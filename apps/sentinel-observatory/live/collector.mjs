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
 * Workers fetch cannot directly request IP URLs; those return an honest limit.
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
  if (url.port && !["80", "443"].includes(url.port)) fail("unsupported_port", "Only ports 80 and 443 can be observed.");
  const blocked = options.blockedHosts || (options.observatoryHost ? [options.observatoryHost] : []);
  const kind = verifyHost(url.hostname, blocked);
  if (kind === "hostname") url.hostname = canonicalHost(url.hostname);
  if (url.pathname.length > 512) fail("path_too_long", "The public path must be at most 512 characters.");
  const normalized = url.toString();
  return { id: stableTargetId(normalized), url: normalized, label: `${url.host}${url.pathname === "/" ? "" : url.pathname}`, hostname: canonicalHost(url.hostname), kind };
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
      method: "GET", redirect: "error", signal,
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
  return null;
}

function safeDiagnostic(error, stage, classifier) {
  const allowedNames = ["Error", "TypeError", "RangeError", "SyntaxError", "AbortError", "TimeoutError", "DOMException"];
  const exceptionName = allowedNames.includes(error?.name) ? error.name : "Error";
  console.error("sentinel_collector_failure", JSON.stringify({ stage, classifier: classifier || "transport_failure", exception_name: exceptionName }));
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
  if (target.kind !== "hostname") {
    fail("platform_ip_unsupported", "Direct IP monitoring needs the local collector. Run python apps/sentinel-observatory/live/local_server.py and open http://127.0.0.1:8766.", 422);
  }
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
    target: target.label, url: target.url, observed_at: isoNow(), completed_at: null,
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
  try {
    if (controller.signal.aborted) throw new DOMException("Cancelled", "AbortError");
    stage = "dns_preflight";
    event.coverage.dns = await validateDns(target, fetchImpl, controller.signal, blockedHosts);
    if (controller.signal.aborted) throw new DOMException("Cancelled", "AbortError");
    event.observed_at = isoNow();
    headStart = monotonic();
    event.coverage.head_attempted = true;
    stage = "head_request";
    // New headers only: never forward incoming authentication, cookies, client IP,
    // Origin, arbitrary headers, or Cloudflare Access credentials to the target.
    const response = await fetchImpl(target.url, {
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
          const next = normalizeTarget(new URL(location, target.url).toString(), { blockedHosts });
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
    if (target.kind !== "hostname") fail("platform_ip_unsupported", "Direct IP monitoring needs the local collector. Run python apps/sentinel-observatory/live/local_server.py and open http://127.0.0.1:8766.", 422);
    const retry = reserveRate(target.url);
    if (retry) return jsonResponse({ error: { code: "sample_rate_limited", message: "Wait before measuring this target again." }, retry_after_ms: retry, rate_limit_scope: "Per isolate and canonical target; not durable or globally coordinated" }, 429, { "Retry-After": String(Math.ceil(retry / 1000)) });
    const result = await collectObservation(target, { blockedHosts, signal: request.signal });
    result.rate_limit_scope = "Per isolate and canonical target; not durable or globally coordinated";
    return jsonResponse(result);
  } catch (error) {
    if (error instanceof ObservationError) return jsonResponse({ error: { code: error.code, message: error.message } }, error.httpStatus);
    return jsonResponse({ error: { code: "invalid_request", message: "The request could not be safely processed." } }, 400);
  }
}
