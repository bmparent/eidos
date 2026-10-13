/** Public literal-IP observer. Never resolves DNS, follows redirects, reads bodies,
 * sends client headers, changes defenses, or retries TLS insecurely. */
import { createHash, randomUUID, timingSafeEqual } from "node:crypto";
import { isIP } from "node:net";
import * as http from "node:http";
import * as https from "node:https";
import { checkServerIdentity } from "node:tls";

export const SAMPLE_INTERVAL_MS = 15_000;
const TIMEOUT_MS = 8_000;
const MAX_REQUEST_BYTES = 1_024;
const SECURITY_KEYS = ["hsts_present", "csp_present", "x_content_type_options", "frame_options", "referrer_policy", "cache_private", "cache_no_store"] as const;

export class ObservationBoundaryError extends Error {
  constructor(readonly code: string, message: string, readonly status = 400) { super(message); }
}
const fail = (code: string, message: string, status = 400): never => { throw new ObservationBoundaryError(code, message, status); };

function ipv4Number(host: string): number {
  return host.split(".").reduce((value, part) => value * 256 + Number(part), 0);
}

export function isPublicIp(host: string): boolean {
  const family = isIP(host);
  if (family === 4) {
    const value = ipv4Number(host);
    const denied: Array<[string, number]> = [
      ["0.0.0.0", 8], ["10.0.0.0", 8], ["100.64.0.0", 10], ["127.0.0.0", 8],
      ["169.254.0.0", 16], ["172.16.0.0", 12], ["192.0.0.0", 24], ["192.0.2.0", 24],
      ["192.88.99.0", 24], ["192.168.0.0", 16], ["198.18.0.0", 15],
      ["198.51.100.0", 24], ["203.0.113.0", 24], ["224.0.0.0", 3],
    ];
    return !denied.some(([base, bits]) => Math.floor(value / 2 ** (32 - bits)) === Math.floor(ipv4Number(base) / 2 ** (32 - bits)));
  }
  if (family !== 6) return false;
  // URL canonicalization expands dotted tails to hex. Global-unicast-only also
  // excludes IPv4 mapped/compatible, NAT64, link local, multicast and ULA.
  const canonical = new URL(`https://[${host}]/`).hostname.slice(1, -1);
  const [left, right = ""] = canonical.split("::");
  const before = left ? left.split(":") : [];
  const after = right ? right.split(":") : [];
  const words = [...before, ...Array(canonical.includes("::") ? 8 - before.length - after.length : 0).fill("0"), ...after].map(word => parseInt(word, 16));
  if ((words[0] & 0xe000) !== 0x2000) return false;
  if (words[0] === 0x2001 && words[1] < 0x0200) return false; // Special use and Teredo.
  if (words[0] === 0x2001 && words[1] === 0x0db8) return false;
  if (words[0] === 0x2002) return false; // 6to4 embeds otherwise forbidden IPv4.
  if (words[0] === 0x3fff && words[1] < 0x1000) return false;
  return true;
}

export interface IpTarget { id: string; url: string; label: string; host: string; scheme: "http:" | "https:"; port: 80 | 443; path: string; }

export function normalizeIpTarget(input: unknown): IpTarget {
  if (typeof input !== "string" || !input.length || input.length > 768) fail("invalid_target", "Enter one public literal IP or an HTTP(S) URL containing that IP.");
  if (input !== input.trim() || /[\u0000-\u0020\u007f\\]/.test(input)) fail("invalid_target", "Whitespace, control characters and backslashes are not allowed.");
  let candidate = input;
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(candidate)) candidate = `https://${isIP(candidate) === 6 ? `[${candidate}]` : candidate}`;
  if (candidate.includes("?") || candidate.includes("#")) fail("query_not_allowed", "Queries and fragments are not collected.");
  let url: URL;
  try { url = new URL(candidate); } catch { return fail("invalid_target", "The literal IP URL could not be parsed."); }
  if (url.protocol !== "http:" && url.protocol !== "https:") fail("unsupported_protocol", "Only HTTP and HTTPS can be observed.");
  if (url.username || url.password) fail("credentials_not_allowed", "Credentials are not allowed in monitoring URLs.");
  // Validate the original authority before WHATWG URL can convert hexadecimal,
  // octal or integer IPv4 aliases. Hostnames and DNS are deliberately excluded.
  const authority = candidate.match(/^[a-z][a-z0-9+.-]*:\/\/([^/]+)/i)?.[1] || "";
  const rawHost = authority.startsWith("[") ? authority.slice(1, authority.indexOf("]")) : authority.split(":")[0];
  if (!isIP(rawHost)) fail("literal_ip_required", "This collector accepts public literal IP addresses only.");
  const host = url.hostname.replace(/^\[|\]$/g, "");
  if (!isPublicIp(host)) fail("non_public_target", "Private, loopback, reserved, mapped and special-use IP ranges cannot be monitored.");
  const port = url.protocol === "https:" ? 443 : 80;
  if (url.port && Number(url.port) !== port) fail("unsupported_port", "Only the scheme's standard port is permitted: HTTP 80 or HTTPS 443.");
  if (url.pathname.length > 512) fail("path_too_long", "The public path must be at most 512 characters.");
  const normalized = url.toString();
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(normalized)) hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 0x100000001b3n);
  return { id: `target_${hash.toString(16).padStart(16, "0")}`, url: normalized,
    label: `${url.host}${url.pathname === "/" ? "" : url.pathname}`, host,
    scheme: url.protocol as IpTarget["scheme"], port, path: url.pathname };
}

function safeHeaders(headers: http.IncomingHttpHeaders) {
  const enumValue = (name: string, allowed: string[]) => {
    const raw = headers[name];
    if (typeof raw !== "string") return null;
    const value = raw.toLowerCase().split(";", 1)[0].trim();
    return allowed.includes(value) ? value : "other";
  };
  return { server: enumValue("server", ["cloudflare", "nginx", "apache", "vercel", "cloudfront"]),
    "content-type": enumValue("content-type", ["text/html", "text/plain", "application/json", "text/css", "application/javascript"]),
    "cf-cache-status": enumValue("cf-cache-status", ["hit", "miss", "dynamic", "bypass", "expired", "stale", "revalidated", "updating"]) };
}

function securityHeaders(headers?: http.IncomingHttpHeaders): Record<string, boolean | string | null> {
  if (!headers) return Object.fromEntries(SECURITY_KEYS.map(key => [key, null]));
  const enumValue = (key: string, allowed: string[]) => {
    const raw = headers[key];
    if (raw === undefined) return "absent";
    const value = typeof raw === "string" ? raw.trim().toLowerCase() : "";
    return allowed.includes(value) ? value : "other";
  };
  const directives = new Set(String(headers["cache-control"] || "").split(",").map(value => value.split("=", 1)[0].trim().toLowerCase()));
  return { hsts_present: Boolean(headers["strict-transport-security"]), csp_present: Boolean(headers["content-security-policy"]),
    x_content_type_options: enumValue("x-content-type-options", ["nosniff"]), frame_options: enumValue("x-frame-options", ["deny", "sameorigin"]),
    referrer_policy: enumValue("referrer-policy", ["no-referrer", "same-origin", "origin", "strict-origin", "origin-when-cross-origin", "strict-origin-when-cross-origin", "unsafe-url", "no-referrer-when-downgrade"]),
    cache_private: directives.has("private"), cache_no_store: directives.has("no-store") };
}

type RequestFactory = (options: https.RequestOptions, callback: (response: http.IncomingMessage) => void) => http.ClientRequest;
interface ObserveOptions { requestFactory?: RequestFactory; timeoutMs?: number; signal?: AbortSignal; }

/** Node connects to the validated numeric hostname. Custom lookup also fails
 * closed if a future transport unexpectedly attempts DNS. Test injection is
 * code-only and is never selected by request JSON or environment values. */
export async function observeIp(input: IpTarget | string, options: ObserveOptions = {}) {
  const target = normalizeIpTarget(typeof input === "string" ? input : input.url);
  const start = performance.now();
  const event = {
    id: `evt_${randomUUID().replaceAll("-", "")}`, source_id: target.id, target: target.label, url: target.url,
    observed_at: new Date().toISOString(), completed_at: null as string | null, latency_ms: null as number | null,
    elapsed_ms: 0, method: "HEAD", kind: "measurement", status: null as number | null,
    error: null as { code: string; message: string; scope: string } | null,
    headers: {} as Record<string, string | null>, security_headers: securityHeaders(), redirect_count: 0,
    redirect_present: false, response_chain_complete: false,
    redirect: null as { followed: boolean; target: string | null; validation: string } | null,
    coverage: { vantage: "Node public-only pinned literal-IP HTTP client", sampling: "One requested HEAD; no crawling or traffic capture",
      head_attempted: false, redirect_followed: false, body_collected: false,
      dns: { state: "literal_public_ip_pinned", address_pinned: true, resolver: "none: literal IP" },
      missing: ["Visitor traffic", "Origin logs", "Internal network", "Packet capture", "Certificate details", "Authenticated application state"],
      dns_rebinding: "No DNS lookup occurs; the validated numeric IP is the connection destination.",
      protections: "Observational only; no network protections or defensive actions changed" },
  };
  try {
    if (options.signal?.aborted) throw Object.assign(new Error(), { code: "OBSERVATION_CANCELLED" });
    await new Promise<void>((resolve, reject) => {
      let request: http.ClientRequest | undefined;
      let settled = false;
      const done = (error?: unknown) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        options.signal?.removeEventListener("abort", cancel);
        if (error) reject(error); else resolve();
      };
      const cancel = () => {
        const error = Object.assign(new Error(), { code: "OBSERVATION_CANCELLED" });
        request?.destroy(error);
        done(error);
      };
      const timer = setTimeout(() => {
        const error = Object.assign(new Error(), { code: "OBSERVATION_TIMEOUT" });
        request?.destroy(error);
        done(error);
      }, options.timeoutMs ?? TIMEOUT_MS);
      try {
        const requestFactory = options.requestFactory ?? (target.scheme === "https:" ? https.request : http.request);
        request = requestFactory({ protocol: target.scheme, hostname: target.host, port: target.port, path: target.path,
          method: "HEAD", agent: false, maxHeaderSize: 16_384, rejectUnauthorized: true,
          // An IP has no DNS SNI name. Certificate validation remains bound to
          // this exact IP via checkServerIdentity, including IP SAN matching.
          servername: "", checkServerIdentity: (_name, cert) => checkServerIdentity(target.host, cert),
          lookup: (_hostname, _lookupOptions, callback) => callback(Object.assign(new Error(), { code: "OBSERVATION_DNS_FORBIDDEN" }), "", 0),
          headers: { Host: isIP(target.host) === 6 ? `[${target.host}]` : target.host,
            "User-Agent": "Eidos-Sentinel-Observatory/0.2 (authorized read-only HEAD)",
            Accept: "*/*", Connection: "close", "Cache-Control": "no-cache", Pragma: "no-cache" },
        }, response => {
          // Do not read or resume response bodies, even when HEAD is violated.
          // Node's http/https APIs do not follow redirects.
          const status = response.statusCode;
          if (status === undefined || status < 100 || status > 599) {
            response.destroy();
            done(Object.assign(new Error(), { code: "OBSERVATION_INVALID_RESPONSE" }));
            return;
          }
          event.status = status;
          event.headers = safeHeaders(response.headers);
          event.security_headers = securityHeaders(response.headers);
          event.redirect_present = status >= 300 && status < 400 && Boolean(response.headers.location);
          event.response_chain_complete = !event.redirect_present;
          if ([301, 302, 303, 307, 308].includes(status)) {
            event.redirect = { followed: false, target: null, validation: "not_available" };
            const location = response.headers.location;
            if (typeof location === "string" && location.length <= 768) {
              try {
                const next = normalizeIpTarget(new URL(location, target.url).toString());
                event.redirect = { followed: false, target: next.url, validation: "syntax_public_literal_only_not_requested" };
              } catch { event.redirect.validation = "withheld_outside_boundary"; }
            }
          }
          response.destroy();
          done();
        });
        request.once("error", done);
        options.signal?.addEventListener("abort", cancel, { once: true });
        if (options.signal?.aborted) { cancel(); return; }
        event.coverage.head_attempted = true;
        event.observed_at = new Date().toISOString();
        request.end();
      } catch (error) { request?.destroy(); done(error); }
    });
  } catch (error) {
    // Use fixed classifications only. Never return/log transport messages, raw
    // certificates, arbitrary response headers, cookies or authorization.
    const code = typeof error === "object" && error !== null && "code" in error ? String(error.code) : "";
    const tlsError = /^(?:ERR_TLS_|ERR_SSL_|CERT_|DEPTH_ZERO_SELF_SIGNED_CERT$|SELF_SIGNED_CERT_IN_CHAIN$|UNABLE_TO_)/.test(code);
    event.error = code === "OBSERVATION_TIMEOUT" ? { code: "timeout", message: "The observation deadline elapsed.", scope: "observation_vantage" }
      : code === "OBSERVATION_CANCELLED" ? { code: "cancelled", message: "The observation was cancelled.", scope: "observation_vantage" }
      : tlsError ? { code: "tls_error", message: "TLS validation or negotiation failed; no insecure retry was made.", scope: "observation_vantage" }
      : { code: "transport_error", message: "The observer did not obtain a usable response; target availability is unknown.", scope: "observation_vantage" };
  } finally {
    event.completed_at = new Date().toISOString();
    event.elapsed_ms = Math.round((performance.now() - start) * 1_000) / 1_000;
    event.latency_ms = event.coverage.head_attempted ? event.elapsed_ms : null;
  }
  return { event, target: { id: target.id, url: target.url, label: target.label }, retry_after_ms: SAMPLE_INTERVAL_MS };
}

const json = (value: unknown, status = 200, extra: Record<string, string> = {}) => Response.json(value, { status, headers: {
  "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", ...extra,
} });

async function boundedJson(request: Request): Promise<unknown> {
  const declared = request.headers.get("content-length");
  if (declared !== null && (!/^\d+$/.test(declared) || Number(declared) > MAX_REQUEST_BYTES)) fail("payload_too_large", "The request must be at most 1024 bytes.", 413);
  if (!request.body) fail("invalid_json", "Send a JSON object containing only target.");
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  const timer = setTimeout(() => { void reader.cancel(); }, 2_000);
  const deadline = performance.now() + 2_000;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (performance.now() >= deadline) fail("request_timeout", "The request body deadline elapsed.", 408);
      if (done) break;
      length += value.byteLength;
      if (length > MAX_REQUEST_BYTES) { await reader.cancel(); fail("payload_too_large", "The request must be at most 1024 bytes.", 413); }
      chunks.push(value);
    }
    const bytes = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    try { return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)); }
    catch { return fail("invalid_json", "Send valid JSON containing only target."); }
  } finally { clearTimeout(timer); reader.releaseLock(); }
}

export function createIpObservationHandler(options: {
  getSecret?: () => string | undefined;
  collector?: (target: IpTarget, signal: AbortSignal) => Promise<Awaited<ReturnType<typeof observeIp>>>;
  clock?: () => number;
} = {}) {
  const rates = new Map<string, number>();
  let active = 0;
  const clock = options.clock ?? (() => performance.now());
  return async (request: Request): Promise<Response> => {
    const secret = (options.getSecret ?? (() => process.env.OBSERVATORY_COLLECTOR_SECRET))();
    // Server-to-server bearer secret. It must never be put in browser bundles.
    if (!secret || secret.length < 32 || secret.length > 512 || /\s/.test(secret)) return json({ error: { code: "collector_unconfigured", message: "The IP collector is not configured." } }, 503);
    const authorization = request.headers.get("authorization") || "";
    if (authorization.length > 1_024 || !timingSafeEqual(createHash("sha256").update(authorization).digest(), createHash("sha256").update(`Bearer ${secret}`).digest())) {
      return json({ error: { code: "unauthorized", message: "Collector authentication is required." } }, 401);
    }
    if (request.method !== "POST") return json({ error: { code: "method_not_allowed", message: "Use POST to request one observation." } }, 405, { Allow: "POST" });
    if (request.headers.get("content-type")?.split(";", 1)[0].trim().toLowerCase() !== "application/json") return json({ error: { code: "json_required", message: "Send a small JSON object containing only target." } }, 415);
    let reserved = false;
    try {
      const payload = await boundedJson(request);
      if (!payload || typeof payload !== "object" || Array.isArray(payload) || Object.keys(payload).length !== 1 || !("target" in payload)) fail("invalid_payload", "The request must contain exactly one target field.");
      const target = normalizeIpTarget(payload.target);
      const now = clock();
      for (const [key, at] of rates) if (now - at >= SAMPLE_INTERVAL_MS) rates.delete(key);
      const wait = SAMPLE_INTERVAL_MS - (now - (rates.get(target.host) ?? -Infinity));
      if (wait > 0) return json({ error: { code: "cooldown", message: "Wait at least 15 seconds between checks of this IP." }, retry_after_ms: Math.ceil(wait) }, 429, { "Retry-After": String(Math.ceil(wait / 1_000)) });
      if (active >= 4 || rates.size >= 512) return json({ error: { code: "busy", message: "The collector is at capacity; try again later." }, retry_after_ms: SAMPLE_INTERVAL_MS }, 429, { "Retry-After": "15" });
      rates.set(target.host, now);
      active += 1;
      reserved = true;
      const result = await (options.collector ?? ((value, signal) => observeIp(value, { signal })))(target, request.signal);
      return json({ ...result, rate_limit_scope: "this Node instance; at most four active observations and one HEAD per IP every 15 seconds" });
    } catch (error) {
      if (error instanceof ObservationBoundaryError) return json({ error: { code: error.code, message: error.message } }, error.status);
      return json({ error: { code: "collector_error", message: "The collector could not complete this request." } }, 500);
    } finally { if (reserved) active -= 1; }
  };
}
