/**
 * One observational HTTP HEAD per accepted user request. Cloudflare Workers,
 * standard Web APIs only; never use a VPC/service binding as the fetch transport.
 *
 * Deployment boundary: owner authentication belongs to the dispatch layer. This
 * handler requires exact same-origin POST but Origin is not authentication.
 * Deploy as a standalone Worker with no privileged origin or private bindings.
 *
 * DNS-over-HTTPS preflight is a filter, NOT address pinning: Worker fetch resolves
 * the hostname independently. We do not claim complete DNS-rebinding protection.
 * Workers fetch cannot directly request IP URLs. A configured authenticated Node
 * collector pins literal IPs; the fixed 1.1.1.1 demo can use an explicitly named
 * DNS service adapter, which is never represented as an IP-pinned measurement.
 * Docs checked 2026-10-13:
 * https://developers.cloudflare.com/workers/platform/known-issues/#fetch-to-ip-addresses
 * https://developers.cloudflare.com/workers/runtime-apis/request/#the-cf-property-requestinitcfproperties
 * https://developers.cloudflare.com/1.1.1.1/encryption/dns-over-https/make-api-requests/dns-json/
 */

export const SAMPLE_INTERVAL_MS = 15_000;
const FETCH_TIMEOUT_MS = 8_000;
const MAX_REQUEST_BYTES = 1_024;
const MAX_DNS_BYTES = 16_384;
const MAX_RATE_KEYS = 512;
const rates = new Map();
const SECURITY_KEYS = ["hsts_present", "csp_present", "x_content_type_options", "frame_options", "referrer_policy", "cache_private", "cache_no_store"];
const BOUNDED_HEADERS = ["cf-cache-status", "cache-status", "server", "content-type", "date"];
const DNS_ENDPOINT = "https://cloudflare-dns.com/dns-query";
const IP_SERVICE_HOST = "one.one.one.one";
const MISSING = ["Visitor traffic", "Origin logs", "Internal network", "Packet capture", "Certificate inspection", "Authenticated application state"];

class ObservationError extends Error {
  constructor(code, message, httpStatus = 400) {
    super(message);
    this.code = code;
    this.httpStatus = httpStatus;
  }
}

const fail = (code, message, status) => { throw new ObservationError(code, message, status); };
const isoNow = () => new Date().toISOString();
const monotonic = () => performance.now();
const isRedirect = (status) => [301, 302, 303, 307, 308].includes(status);

async function discardBody(response) {
  // Cleanup failure cannot erase an HTTP status/headers already observed.
  try { await response.body?.cancel(); } catch { /* No body read or logging. */ }
}

function ipv4Number(host) {
  if (!/^(?:\d{1,3}\.){3}\d{1,3}$/.test(host)) return null;
  const parts = host.split(".").map(Number);
  if (parts.some((part) => part > 255)) return null;
  return parts.reduce((value, part) => value * 256 + part, 0);
}

function publicIPv4(host) {
  const value = ipv4Number(host);
  if (value === null) return false;
  // Conservative exclusions include special-use, documentation, benchmark,
  // multicast and reserved ranges, not only RFC1918.
  const denied = [
    ["0.0.0.0", 8], ["10.0.0.0", 8], ["100.64.0.0", 10], ["127.0.0.0", 8],
    ["169.254.0.0", 16], ["172.16.0.0", 12], ["192.0.0.0", 24], ["192.0.2.0", 24],
    ["192.88.99.0", 24], ["192.168.0.0", 16], ["198.18.0.0", 15],
    ["198.51.100.0", 24], ["203.0.113.0", 24], ["224.0.0.0", 3],
  ];
  return !denied.some(([base, bits]) => Math.floor(value / 2 ** (32 - bits)) === Math.floor(ipv4Number(base) / 2 ** (32 - bits)));
}

function ipv6Words(host) {
  const raw = host.replace(/^\[|\]$/g, "").toLowerCase();
  if (!/^[0-9a-f:]+$/.test(raw) || raw.includes(":::")) return null;
  const halves = raw.split("::");
  if (halves.length > 2) return null;
  const left = halves[0] ? halves[0].split(":") : [];
  const right = halves.length === 2 && halves[1] ? halves[1].split(":") : [];
  if ([...left, ...right].some((word) => !/^[0-9a-f]{1,4}$/.test(word))) return null;
  const gap = 8 - left.length - right.length;
  if (halves.length === 1 && gap !== 0 || halves.length === 2 && gap < 1) return null;
  return [...left, ...Array(halves.length === 2 ? gap : 0).fill("0"), ...right].map((word) => parseInt(word, 16));
}

function publicIPv6(host) {
  const words = ipv6Words(host);
  if (!words || words.length !== 8 || (words[0] & 0xe000) !== 0x2000) return false;
  // Global-unicast only; exclude Teredo/special 2001::/23, documentation,
  // 6to4 embedded-address tunnels and the newer 3fff::/20 documentation block.
  if (words[0] === 0x2001 && words[1] < 0x0200) return false;
  if (words[0] === 0x2001 && words[1] === 0x0db8) return false;
  if (words[0] === 0x2002) return false;
  if (words[0] === 0x3fff && words[1] < 0x1000) return false;
  return true;
}

function canonicalHost(host) {
  return host.toLowerCase().replace(/\.$/, "").replace(/^\[|\]$/g, "");
}

function verifyHost(host, blockedHosts = []) {
  const canonical = canonicalHost(host);
  if (blockedHosts.some((blocked) => canonicalHost(blocked) === canonical)) {
    fail("recursive_target", "The observatory cannot monitor its own service hostname.");
  }
  if (canonical.includes(":")) {
    if (!publicIPv6(canonical)) fail("non_public_target", "Only public global unicast IP addresses can be selected.");
    return "ipv6";
  }
  if (ipv4Number(canonical) !== null) {
    if (!publicIPv4(canonical)) fail("non_public_target", "Private, loopback, reserved, and special-use IP ranges cannot be monitored.");
    return "ipv4";
  }
  if (canonical.length > 253 || !canonical.includes(".") || !/^[a-z0-9.-]+$/.test(canonical)) {
    fail("invalid_hostname", "Enter a public fully qualified hostname or public IP address.");
  }
  const labels = canonical.split(".");
  if (labels.some((label) => !/^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(label))) {
    fail("invalid_hostname", "The hostname is not valid.");
  }
  if (labels.at(-1).match(/^\d+$/) || /(?:^|\.)(?:localhost|local|internal|intranet|lan|home|localdomain|test|invalid|example|onion|alt|arpa)$/.test(canonical)) {
    fail("non_public_target", "Local, private, and reserved DNS names cannot be monitored.");
  }
  return "hostname";
}

function stableTargetId(value) {
  // Identity only, not an evidence checksum or security signature.
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(value)) hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 0x100000001b3n);
  return `target_${hash.toString(16).padStart(16, "0")}`;
}

/** Pure, synchronous normalization. DNS validation happens before any HEAD. */
export function normalizeTarget(input, options = {}) {
  if (typeof input !== "string" || input.length < 1 || input.length > 768) fail("invalid_target", "Enter a URL or public IP address of at most 768 characters.");
  const trimmed = input.trim();
  if (!trimmed || /[\u0000-\u0020\u007f\\]/.test(trimmed)) fail("invalid_target", "Whitespace, control characters, and backslashes are not allowed in targets.");
  let candidate = trimmed;
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(candidate)) {
    if (/^[0-9a-f:]+$/i.test(candidate) && candidate.includes(":")) candidate = `[${candidate}]`;
    candidate = `https://${candidate}`;
  }
  let url;
  try { url = new URL(candidate); } catch { fail("invalid_target", "The target could not be parsed as a URL."); }
  if (!["http:", "https:"].includes(url.protocol)) fail("unsupported_protocol", "Only HTTP and HTTPS can be observed.");
  if (url.username || url.password) fail("credentials_not_allowed", "Credentials are not allowed in a monitoring URL.");
  // Test original delimiter presence as empty query/fragment is still excluded.
  if (candidate.includes("?") || candidate.includes("#")) fail("query_not_allowed", "Queries and fragments are not collected; choose a fixed public path.");
  if (url.port && url.port !== (url.protocol === "https:" ? "443" : "80")) fail("unsupported_port", "Use the standard port for the scheme: HTTP 80 or HTTPS 443.");
  const blocked = options.blockedHosts || (options.observatoryHost ? [options.observatoryHost] : []);
  const kind = verifyHost(url.hostname, blocked);
  if (kind === "hostname") url.hostname = canonicalHost(url.hostname);
  if (url.pathname.length > 512) fail("path_too_long", "The public path must be at most 512 characters.");
  const normalized = url.toString();
  return { id: stableTargetId(normalized), url: normalized, label: `${url.host}${url.pathname === "/" ? "" : url.pathname}`, hostname: canonicalHost(url.hostname), kind };
}

function serviceAdapter(target, blockedHosts) {
  if (target.kind === "hostname") return target;
  // The well-known public resolver is a service test, not an arbitrary reverse
  // DNS lookup. It preserves the path and scheme, and TLS authenticates the
  // named service. Other literal IPs require the actual pinned-address relay.
  if (target.hostname !== "1.1.1.1") fail("platform_ip_unsupported", "Direct IP monitoring requires the configured public IP collector, or the local companion: python apps/sentinel-observatory/live/local_server.py. The hosted 1.1.1.1 test uses the explicitly named one.one.one.one service.", 422);
  const url = new URL(target.url);
  url.hostname = IP_SERVICE_HOST;
  return normalizeTarget(url.toString(), { blockedHosts });
}

/** Configuration only, never accepted from browser request JSON. */
export function ipCollectorConfiguration(env = {}) {
  const raw = typeof env.OBSERVATORY_COLLECTOR_URL === "string" ? env.OBSERVATORY_COLLECTOR_URL.trim() : "";
  const secret = typeof env.OBSERVATORY_COLLECTOR_SECRET === "string" ? env.OBSERVATORY_COLLECTOR_SECRET.trim() : "";
  if (!raw || !/^[a-f0-9]{64}$/.test(secret)) return null;
  try {
    const url = new URL(raw);
    if (url.protocol !== "https:" || !url.hostname.endsWith(".vercel.app") || url.port || url.username || url.password || url.search || url.hash || url.pathname !== "/api/observatory/ip-observe") return null;
    verifyHost(url.hostname);
    const bypass = typeof env.OBSERVATORY_COLLECTOR_BYPASS_TOKEN === "string" ? env.OBSERVATORY_COLLECTOR_BYPASS_TOKEN.trim() : "";
    if (bypass.length > 1024 || /[\u0000-\u0020\u007f]/.test(bypass)) return null;
    return { url: url.toString(), secret, bypass };
  } catch { return null; }
}

async function boundedText(response, limit, errorCode) {
  const declared = response.headers.get("content-length");
  if (declared !== null && (!/^\d+$/.test(declared) || Number(declared) > limit)) {
    await response.body?.cancel();
    fail(errorCode, "The response exceeded the permitted size.", 400);
  }
  if (!response.body) return "";
  const reader = response.body.getReader();
  const chunks = [];
  let length = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > limit) {
        await reader.cancel();
        fail(errorCode, "The response exceeded the permitted size.", 400);
      }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
}

async function validateDns(target, fetchImpl, signal, blockedHosts) {
  const responses = await Promise.all(["A", "AAAA"].map(async (type) => {
    const url = new URL(DNS_ENDPOINT);
    url.searchParams.set("name", target.hostname);
    url.searchParams.set("type", type);
    const response = await fetchImpl(url.toString(), {
      // Older edge runtimes reject redirect:"error" before any I/O. Manual
      // mode plus the non-2xx check below rejects redirects without following.
      method: "GET", redirect: "manual", signal,
      headers: { Accept: "application/dns-json", "Cache-Control": "no-cache", Pragma: "no-cache" },
      cf: { cacheTtlByStatus: { "100-599": -1 } },
    });
    if (!response.ok || response.redirected) {
      await response.body?.cancel();
      fail("dns_unavailable", "The public DNS preflight could not complete; no target request was sent.", 502);
    }
    let payload;
    try { payload = JSON.parse(await boundedText(response, MAX_DNS_BYTES, "dns_invalid_response")); }
    catch (error) {
      if (error instanceof ObservationError) throw error;
      fail("dns_invalid_response", "The public DNS preflight returned unusable data; no target request was sent.", 502);
    }
    if (payload.Status !== 0 || payload.TC === true || payload.Answer !== undefined && !Array.isArray(payload.Answer)) {
      fail("dns_unavailable", "The public DNS preflight did not establish usable public addresses; no target request was sent.", 502);
    }
    const answer = payload.Answer || [];
    if (answer.length > 32) fail("dns_invalid_response", "The public DNS answer exceeded the permitted record count.", 502);
    return answer;
  }));
  const addresses = new Set();
  for (const record of responses.flat()) {
    if (record.type === 5) {
      if (typeof record.data !== "string") fail("dns_invalid_response", "The public DNS alias was invalid.", 502);
      verifyHost(record.data, blockedHosts);
    }
    if (record.type === 1 || record.type === 28) {
      if (typeof record.data !== "string") fail("dns_invalid_response", "The public DNS address was invalid.", 502);
      const publicAddress = record.type === 1 ? publicIPv4(record.data) : publicIPv6(record.data);
      if (!publicAddress) fail("dns_non_public", "DNS returned a private, reserved, or non-global address. No target request was sent.", 400);
      addresses.add(record.data.toLowerCase());
    }
  }
  if (!addresses.size) fail("dns_no_address", "Public DNS returned no usable address; no target request was sent.", 502);
  return { address_count: addresses.size, resolver: "Cloudflare DNS over HTTPS", state: "public_addresses_preflight", address_pinned: false };
}

function securityHeaders(headers) {
  if (!headers) return Object.fromEntries(SECURITY_KEYS.map((key) => [key, null]));
  const enumHeader = (key, allowed) => {
    const value = headers.get(key);
    if (value === null) return "absent";
    const normalized = value.trim().toLowerCase();
    return allowed.includes(normalized) ? normalized : "other";
  };
  const cache = new Set((headers.get("cache-control") || "").split(",").map((part) => part.trim().split("=", 1)[0].toLowerCase()));
  return {
    hsts_present: Boolean(headers.get("strict-transport-security")),
    csp_present: Boolean(headers.get("content-security-policy")),
    x_content_type_options: enumHeader("x-content-type-options", ["nosniff"]),
    frame_options: enumHeader("x-frame-options", ["deny", "sameorigin"]),
    referrer_policy: enumHeader("referrer-policy", ["no-referrer", "no-referrer-when-downgrade", "origin", "origin-when-cross-origin", "same-origin", "strict-origin", "strict-origin-when-cross-origin", "unsafe-url"]),
    cache_private: cache.has("private"), cache_no_store: cache.has("no-store"),
  };
}

function runtimeClassifier(error) {
  // Inspect only to select fixed codes; never return/log the raw message.
  const message = typeof error?.message === "string" ? error.message.slice(0, 512) : "";
  if (/illegal invocation|incorrect this reference/i.test(message)) return "illegal_invocation";
  if (/unsupported cache mode|cache (?:mode|option).{0,40}not supported/i.test(message)) return "unsupported_cache_mode";
  if (/\bcache\b.{0,80}RequestInitializerDict.{0,50}not implemented/i.test(message)) return "cache_option_unimplemented";
  if (/invalid redirect value|unsupported redirect mode/i.test(message)) return "unsupported_redirect_mode";
  return null;
}

function safeDiagnostic(error, stage, classifier) {
  const allowedNames = ["Error", "TypeError", "RangeError", "SyntaxError", "AbortError", "TimeoutError", "DOMException"];
  const exceptionName = allowedNames.includes(error?.name) ? error.name : "Error";
  let message = typeof error?.message === "string" ? error.message.slice(0, 2048) : "No diagnostic message";
  // Server-side diagnostics only. Avoid printing request targets, credentials,
  // opaque identifiers, or arbitrary quoted values even when a runtime embeds
  // them in its exception. Known intrinsic names remain useful for debugging.
  const intrinsicNames = new Set(["cache", "RequestInitializerDict", "URL", "URLSearchParams", "searchParams", "set", "get", "fetch", "signal", "AbortController", "headers", "bind", "toString", "url", "object", "function", "undefined", "manual", "error", "no-store", "no-cache"]);
  message = message
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .replace(/\b(?:authorization|proxy-authorization|cookie|set-cookie|password|passwd|api[_-]?key|access[_-]?token|token|secret)\b\s*[:=]\s*[^,;]+/gi, "[credential redacted]")
    .replace(/\b(?:Bearer|Basic)\s+[^\s,;]+/gi, "[credential redacted]")
    .replace(/\b(?:https?|wss?|ftp):\/\/[^\s\"'<>`]+/gi, "[URL redacted]")
    .replace(/\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?\b/g, "[IP redacted]")
    .replace(/\[?(?:[0-9a-f]{0,4}:){2,}[0-9a-f:.]{0,45}\]?(?:%[a-z0-9]+)?/gi, "[IP redacted]")
    .replace(/\b[a-z0-9-]+(?:\.[a-z0-9-]+)+\b/gi, (value) => /^(?:url|URL|globalThis|controller|signal|response|request|reader)\.(?:searchParams\.)?(?:searchParams|set|get|fetch|bind|toString|signal|headers|read|cancel|addEventListener|removeEventListener|throwIfAborted)$/.test(value) ? value : "[hostname redacted]")
    .replace(/\blocalhost\b/gi, "[hostname redacted]")
    .replace(/\b(?:sk-[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_]+|AKIA[A-Z0-9]+)\b/g, "[token redacted]")
    .replace(/\b(?:secret|token)[a-z0-9_-]*\b/gi, "[sensitive value redacted]")
    .replace(/\b(?=[A-Za-z0-9_+\/=\-]{20,}\b)(?=[A-Za-z0-9_+\/=\-]*[0-9_+\/=\-])[A-Za-z0-9_+\/=\-]{20,}\b/g, "[opaque value redacted]")
    .replace(/([\"'`])([^\"'`\r\n]*)\1/g, (whole, quote, value) => intrinsicNames.has(value) ? whole : "[quoted value redacted]")
    .replace(/\s+/g, " ")
    .trim().slice(0, 200);
  const knownFunctions = new Set(["validateDns", "collectObservation", "handleObservation", "boundedText", "fetch", "Function.fetch", "Object.fetch"]);
  const stackFunctions = typeof error?.stack === "string" ? [...error.stack.slice(0, 2048).matchAll(/\bat\s+(?:async\s+)?([A-Za-z_$][\w$]*(?:\.[\w$]+)*)/g)].map((match) => match[1]).filter((name) => knownFunctions.has(name)).slice(0, 4) : [];
  console.error("sentinel_collector_failure", JSON.stringify({ stage, classifier: classifier || "transport_failure", exception_name: exceptionName, message, stack_functions: stackFunctions }));
}

function redactedError(error, signal, timedOut) {
  if (timedOut) return { code: "timeout", message: "The observation did not complete within eight seconds; site availability is unknown.", scope: "observation_vantage" };
  if (signal?.aborted) return { code: "cancelled", message: "The observation was cancelled before completion.", scope: "observation_vantage" };
  if (error instanceof ObservationError) return { code: error.code, message: error.message, scope: error.code.startsWith("dns_") ? "dns_preflight" : "observation_boundary" };
  const classifier = runtimeClassifier(error);
  if (classifier) return { code: "collector_runtime_error", classifier, message: "The collector hit a runtime compatibility error before completing this observation. Site availability is unknown.", scope: "collector_runtime" };
  return { code: "transport_error", message: "The observation path failed before an HTTP response was available; site availability is unknown.", scope: "observation_vantage" };
}

/**
 * Observe a normalized or raw target. options.fetchImpl exists for deterministic
 * tests, never from request JSON or user-selected bindings.
 */
export async function collectObservation(input, options = {}) {
  const blockedHosts = options.blockedHosts || [];
  const target = normalizeTarget(typeof input === "string" ? input : input?.url, { blockedHosts });
  const transportTarget = serviceAdapter(target, blockedHosts);
  const adapted = transportTarget.url !== target.url;
  // Workers runtime methods can require their receiver. Never detach fetch and
  // invoke it with undefined `this`; Node's permissive fetch hid this in mocks.
  const fetchImpl = options.fetchImpl || globalThis.fetch.bind(globalThis);
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, FETCH_TIMEOUT_MS);
  const cancel = () => controller.abort();
  if (options.signal?.aborted) controller.abort();
  options.signal?.addEventListener("abort", cancel, { once: true });
  const start = monotonic();
  let headStart = null;
  let stage = "initialization";
  const event = {
    id: `evt_${crypto.randomUUID().replaceAll("-", "")}`, source_id: target.id,
    target: adapted ? `${target.label} via ${IP_SERVICE_HOST}` : target.label,
    url: transportTarget.url, observed_at: isoNow(), completed_at: null,
    latency_ms: null, status: null, kind: "measurement", method: "HEAD", error: null,
    headers: {}, security_headers: securityHeaders(null), redirect: null,
    redirect_count: 0, response_chain_complete: false,
    coverage: {
      vantage: "Cloudflare Worker public HTTP subrequest", sampling: "One requested HEAD; no crawling or traffic capture",
      head_attempted: false, redirect_followed: false, body_collected: false,
      dns: { state: "not_checked", address_pinned: false }, missing: MISSING,
      dns_rebinding: "DNS is preflight-checked but not pinned to the later fetch. This is not complete rebinding protection.",
      protections: "Observational only; no network protections or defensive actions changed",
    },
  };
  if (adapted) event.coverage.transport = {
    kind: "named_service_adapter", requested_url: target.url, effective_url: transportTarget.url,
    requested_address: target.hostname, tls_server_name: new URL(transportTarget.url).protocol === "https:" ? IP_SERVICE_HOST : null,
    direct_ip: false, address_pinned: false,
    limitation: "The Worker measures the DNS-routed one.one.one.one service. It does not establish a connection to the literal 1.1.1.1 address.",
  };
  try {
    if (controller.signal.aborted) throw new DOMException("Cancelled", "AbortError");
    stage = "dns_preflight";
    event.coverage.dns = await validateDns(transportTarget, fetchImpl, controller.signal, blockedHosts);
    if (controller.signal.aborted) throw new DOMException("Cancelled", "AbortError");
    event.observed_at = isoNow();
    headStart = monotonic();
    event.coverage.head_attempted = true;
    stage = "head_request";
    // New headers only: never forward incoming authentication, cookies, client IP,
    // Origin, arbitrary headers, or Cloudflare Access credentials to the target.
    const response = await fetchImpl(transportTarget.url, {
      // RequestInit.cache is gated behind Cloudflare compatibility 2024-11-11.
      // Use explicit cache-control headers and the long-supported cf override
      // without requiring a platform compatibility configuration change.
      method: "HEAD", redirect: "manual", signal: controller.signal,
      headers: { "User-Agent": "Eidos-Sentinel-Observatory/0.2 (authorized read-only HEAD monitor)", Accept: "*/*", "Cache-Control": "no-cache", Pragma: "no-cache" },
      cf: { cacheTtlByStatus: { "100-599": -1 } },
    });
    event.latency_ms = Math.round((monotonic() - headStart) * 1000) / 1000;
    if (response.redirected) {
      await discardBody(response);
      fail("unexpected_redirect_follow", "The transport followed a redirect contrary to the observation policy; no final response is attributed to the selected target.", 502);
    }
    if (response.status < 100 || response.status > 599) {
      await discardBody(response);
      fail("unusable_http_response", "No usable HTTP response status was available from the observation path.", 502);
    }
    event.status = response.status;
    stage = "response_metadata";
    event.response_chain_complete = !isRedirect(response.status);
    for (const key of BOUNDED_HEADERS) {
      const value = response.headers.get(key);
      if (value !== null) event.headers[key] = value.replace(/[\u0000-\u001f\u007f]/g, " ").slice(0, 256);
    }
    event.security_headers = securityHeaders(response.headers);
    if (isRedirect(response.status)) {
      const location = response.headers.get("location");
      event.redirect = { followed: false, target: null, validation: "not_available" };
      if (location && location.length <= 768) {
        try {
          const next = normalizeTarget(new URL(location, transportTarget.url).toString(), { blockedHosts });
          event.redirect = { followed: false, target: next.url, validation: "syntax_and_public_literal_only; destination DNS not queried" };
        } catch { event.redirect.validation = "withheld_outside_boundary"; }
      }
    }
    // Deliberately no response.text/json/arrayBuffer: all target bodies are ignored.
    await discardBody(response);
  } catch (error) {
    event.error = redactedError(error, controller.signal, timedOut);
    event.coverage.failure_stage = stage;
    if (!(error instanceof ObservationError) && !controller.signal.aborted && !timedOut) safeDiagnostic(error, stage, runtimeClassifier(error));
    if (headStart !== null && event.latency_ms === null) event.latency_ms = Math.round((monotonic() - headStart) * 1000) / 1000;
  } finally {
    clearTimeout(timer);
    controller.abort(); // Cancel any still-pending DNS sibling after a failed preflight.
    options.signal?.removeEventListener("abort", cancel);
    event.completed_at = isoNow();
    event.elapsed_ms = Math.round((monotonic() - start) * 1000) / 1000;
  }
  return { event, target: { id: target.id, url: target.url, label: target.label }, retry_after_ms: SAMPLE_INTERVAL_MS };
}

const RELAY_ERROR_MESSAGES = {
  tls_error: "TLS validation or negotiation failed; no insecure retry was made. Target availability is unknown.",
  timeout: "The public IP observation did not complete within eight seconds; target availability is unknown.",
  cancelled: "The public IP observation was cancelled before completion.",
  transport_error: "The public IP collector did not obtain a usable response; target availability is unknown.",
  dns_non_public: "The public IP collector rejected a non-public address; no target request was sent.",
};

function validatedRelayResult(payload, target, relayUrl, blockedHosts = []) {
  const event = payload?.event;
  const numberOrNull = (value) => value === null || typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 30_000;
  const date = (value) => typeof value === "string" && value.length < 40 && Number.isFinite(Date.parse(value));
  if (!payload || payload.target?.id !== target.id || payload.target?.url !== target.url || !event || event.source_id !== target.id || event.url !== target.url || !/^evt_[a-f0-9]{32}$/.test(event.id || "") || event.method !== "HEAD" || event.kind !== "measurement" || !date(event.observed_at) || !date(event.completed_at) || !numberOrNull(event.latency_ms) || !numberOrNull(event.elapsed_ms) || event.status !== null && (!Number.isInteger(event.status) || event.status < 100 || event.status > 599) || typeof event.coverage?.head_attempted !== "boolean" || event.coverage?.body_collected !== false || event.coverage?.redirect_followed !== false || event.coverage?.dns?.address_pinned !== true) {
    fail("collector_invalid_response", "The public IP collector returned evidence that did not match this target. No result was accepted.", 502);
  }
  let error = null;
  if (event.error !== null) {
    const code = Object.hasOwn(RELAY_ERROR_MESSAGES, event.error?.code) ? event.error.code : "transport_error";
    error = { code, message: RELAY_ERROR_MESSAGES[code], scope: "observation_vantage" };
  }
  if (event.status === null && !error) fail("collector_invalid_response", "The public IP collector did not provide a response or an explicit observation error.", 502);
  if (event.status !== null && error || Date.parse(event.completed_at) < Date.parse(event.observed_at)) fail("collector_invalid_response", "The public IP collector returned inconsistent response evidence.", 502);
  const headers = {};
  for (const key of BOUNDED_HEADERS) {
    if (typeof event.headers?.[key] === "string") headers[key] = event.headers[key].replace(/[\u0000-\u001f\u007f]/g, " ").slice(0, 256);
  }
  const security = securityHeaders(null);
  for (const key of SECURITY_KEYS) {
    const value = event.security_headers?.[key];
    if (typeof value === "boolean" || value === null || ["absent", "other", "nosniff", "deny", "sameorigin", "no-referrer", "no-referrer-when-downgrade", "origin", "origin-when-cross-origin", "same-origin", "strict-origin", "strict-origin-when-cross-origin", "unsafe-url"].includes(value)) security[key] = value;
  }
  let redirect = null;
  if (isRedirect(event.status)) {
    redirect = { followed: false, target: null, validation: "withheld_outside_boundary" };
    if (typeof event.redirect?.target === "string") {
      try { redirect = { followed: false, target: normalizeTarget(event.redirect.target, { blockedHosts }).url, validation: "syntax_and_public_literal_only; destination DNS not queried" }; } catch { /* No redirect request is made. */ }
    }
  }
  const accepted = {
    id: event.id, source_id: target.id, target: target.label, url: target.url,
    observed_at: event.observed_at, completed_at: event.completed_at,
    latency_ms: event.latency_ms, elapsed_ms: event.elapsed_ms, status: event.status,
    kind: "measurement", method: "HEAD", error, headers, security_headers: security,
    redirect, redirect_count: 0, response_chain_complete: event.status !== null && !isRedirect(event.status),
    coverage: {
      vantage: "Vercel Node public-only pinned-address HTTP client",
      sampling: "One requested HEAD; no crawling or traffic capture",
      head_attempted: event.coverage.head_attempted, redirect_followed: false, body_collected: false,
      dns: { state: "literal_public_ip_pinned", address_pinned: true, resolver: "none: literal public IP" },
      transport: { kind: "authenticated_pinned_ip_relay", direct_ip: true, requested_url: target.url, effective_url: target.url, tls_server_name: new URL(target.url).protocol === "https:" ? target.hostname : null, address_pinned: true },
      relay: { hostname: new URL(relayUrl).hostname, evidence_checked_against_requested_target: true },
      missing: MISSING,
      dns_rebinding: "A validated numeric public IP is the transport destination; no target DNS lookup or alias is used.",
      protections: "Observational only; no network protections or defensive actions changed",
    },
  };
  return { event: accepted, target: { id: target.id, url: target.url, label: target.label }, retry_after_ms: SAMPLE_INTERVAL_MS };
}

/** Private server-to-server relay. The browser cannot select its endpoint/key. */
export async function collectIpRelayObservation(input, configuration, options = {}) {
  const target = normalizeTarget(typeof input === "string" ? input : input?.url, { blockedHosts: options.blockedHosts || [] });
  if (target.kind === "hostname") fail("invalid_ip_relay_target", "The public IP collector accepts a literal public IP address.");
  const config = ipCollectorConfiguration({ OBSERVATORY_COLLECTOR_URL: configuration?.url, OBSERVATORY_COLLECTOR_SECRET: configuration?.secret, OBSERVATORY_COLLECTOR_BYPASS_TOKEN: configuration?.bypass });
  if (!config) fail("collector_not_configured", "The authenticated public IP collector is not configured.", 503);
  const fetchImpl = options.fetchImpl || globalThis.fetch.bind(globalThis);
  const controller = new AbortController();
  const cancel = () => controller.abort();
  if (options.signal?.aborted) controller.abort();
  options.signal?.addEventListener("abort", cancel, { once: true });
  const timer = setTimeout(cancel, 10_000);
  try {
    if (controller.signal.aborted) fail("collector_cancelled", "The public IP observation was cancelled before relay dispatch.", 408);
    const headers = { "Content-Type": "application/json", Authorization: `Bearer ${config.secret}`, "Cache-Control": "no-store" };
    if (config.bypass) headers["x-vercel-protection-bypass"] = config.bypass;
    const response = await fetchImpl(config.url, { method: "POST", headers, body: JSON.stringify({ target: target.url }), redirect: "manual", signal: controller.signal, cf: { cacheTtlByStatus: { "100-599": -1 } } });
    if (!response.ok || response.redirected) {
      await discardBody(response);
      if (response.status === 429) fail("sample_rate_limited", "Wait at least fifteen seconds before measuring this public IP again.", 429);
      if ([401, 403].includes(response.status)) fail("collector_auth_failed", "The public IP collector's server authorization or deployment protection could not be verified.", 502);
      fail("collector_unavailable", "The public IP collector did not return usable evidence. Target availability is unknown.", 502);
    }
    const raw = await boundedText(response, 32_768, "collector_invalid_response");
    let payload;
    try { payload = JSON.parse(raw); } catch { fail("collector_invalid_response", "The public IP collector did not return valid evidence JSON.", 502); }
    return validatedRelayResult(payload, target, config.url, options.blockedHosts || []);
  } catch (error) {
    if (error instanceof ObservationError) throw error;
    fail("collector_transport_error", "The public IP collector connection did not complete. Target availability is unknown.", 502);
  } finally {
    clearTimeout(timer);
    controller.abort();
    options.signal?.removeEventListener("abort", cancel);
  }
}

function jsonResponse(value, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", ...extraHeaders } });
}

function reserveRate(key) {
  const now = Date.now();
  const remaining = SAMPLE_INTERVAL_MS - (now - (rates.get(key) ?? -Infinity));
  if (remaining > 0) return Math.ceil(remaining);
  for (const [oldKey, at] of rates) if (now - at >= SAMPLE_INTERVAL_MS) rates.delete(oldKey);
  // Fail closed when capacity is full; do not evict unexpired rate entries.
  if (!rates.has(key) && rates.size >= MAX_RATE_KEYS) return SAMPLE_INTERVAL_MS;
  rates.set(key, now);
  return 0;
}

/** Dispatch layer must enforce owner authentication before calling this handler. */
export async function handleObservation(request, env = {}, _ctx = undefined) {
  try {
    const endpoint = new URL(request.url);
    if (request.method !== "POST") return jsonResponse({ error: { code: "method_not_allowed", message: "Use POST to request one observation." } }, 405, { Allow: "POST" });
    if (request.headers.get("Origin") !== endpoint.origin || ![null, "same-origin"].includes(request.headers.get("Sec-Fetch-Site"))) {
      return jsonResponse({ error: { code: "same_origin_required", message: "An exact same-origin request is required." } }, 403);
    }
    if (request.headers.get("Content-Type")?.split(";", 1)[0].trim().toLowerCase() !== "application/json") {
      return jsonResponse({ error: { code: "json_required", message: "Send a small JSON object containing only target." } }, 415);
    }
    const text = await boundedText(request, MAX_REQUEST_BYTES, "payload_too_large");
    let payload;
    try { payload = JSON.parse(text); } catch { fail("invalid_json", "The request must be valid JSON."); }
    if (!payload || Array.isArray(payload) || typeof payload !== "object" || Object.keys(payload).length !== 1 || !Object.hasOwn(payload, "target")) {
      fail("invalid_payload", "The request must contain exactly one target field.");
    }
    const blockedHosts = [endpoint.hostname];
    // Operator-configured aliases only; never accept this from the request.
    if (typeof env.OBSERVATORY_HOSTS === "string") blockedHosts.push(...env.OBSERVATORY_HOSTS.split(",").map((host) => host.trim()).filter(Boolean));
    const target = normalizeTarget(payload.target, { blockedHosts });
    const ipCollector = target.kind !== "hostname" ? ipCollectorConfiguration(env) : null;
    if (target.kind !== "hostname" && !ipCollector) serviceAdapter(target, blockedHosts);
    const retry = reserveRate(target.url);
    if (retry) return jsonResponse({ error: { code: "sample_rate_limited", message: "Wait before measuring this target again." }, retry_after_ms: retry, rate_limit_scope: "Per isolate and canonical target; not durable or globally coordinated" }, 429, { "Retry-After": String(Math.ceil(retry / 1000)) });
    const result = ipCollector
      ? await collectIpRelayObservation(target, ipCollector, { blockedHosts, signal: request.signal })
      : await collectObservation(target, { blockedHosts, signal: request.signal });
    result.rate_limit_scope = "Per isolate and canonical target; not durable or globally coordinated";
    return jsonResponse(result);
  } catch (error) {
    if (error instanceof ObservationError) {
      if (error.httpStatus === 429) return jsonResponse({ error: { code: error.code, message: error.message }, retry_after_ms: SAMPLE_INTERVAL_MS }, 429, { "Retry-After": "15" });
      return jsonResponse({ error: { code: error.code, message: error.message } }, error.httpStatus);
    }
    return jsonResponse({ error: { code: "invalid_request", message: "The request could not be safely processed." } }, 400);
  }
}
