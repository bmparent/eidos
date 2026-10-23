import { lookup } from "node:dns/promises";
import { request as httpsRequest } from "node:https";
import { isIP } from "node:net";
import { LabError, LIMITS } from "./store";

export function publicAddress(address: string) {
  const value = address.toLowerCase();
  if (isIP(value) === 6) {
    // Only global-unicast IPv6; mapped IPv4 and special-use ranges fail closed.
    // IPv6 transition mechanisms can embed special-use IPv4 destinations.
    // This pilot importer deliberately qualifies IPv4 only.
    return false;
  }
  if (isIP(value) !== 4) return false;
  const [a, b, c] = value.split(".").map(Number);
  return !(a === 0 || a === 10 || a === 127 || a >= 224 || a === 169 && b === 254 ||
    a === 172 && b >= 16 && b <= 31 || a === 192 && (b === 168 || b === 0 || b === 2) ||
    a === 100 && b >= 64 && b <= 127 || a === 198 && (b === 18 || b === 19 || b === 51 && c === 100) ||
    a === 203 && b === 0 && c === 113);
}

export async function validateDestination(value: string, resolver = lookup) {
  let url: URL;
  try { url = new URL(value); } catch { throw new LabError(400, "Enter a valid public HTTPS URL."); }
  if (url.protocol !== "https:" || url.username || url.password || url.port && url.port !== "443" || url.hash || value.length > 2048)
    throw new LabError(400, "Use a public HTTPS URL without credentials, fragments or custom ports.");
  if (url.searchParams.has("token") || url.searchParams.has("key") || url.searchParams.has("signature"))
    throw new LabError(400, "Authenticated sources require a configured connector; do not paste secret URLs.");
  const host = url.hostname.replace(/^\[|\]$/g, "");
  const addresses = await resolver(host, { all: true, verbatim: true }).catch(() => { throw new LabError(400, "The source hostname could not be resolved."); });
  const ipv4 = addresses.filter(a => a.family === 4);
  if (!ipv4.length || ipv4.some(a => !publicAddress(a.address))) throw new LabError(400, "A public IPv4 destination is required; private, local and special-use addresses are blocked.");
  // Only this validated IPv4 address is handed to the pinned connection. A
  // dual-stack public hostname is supported without enabling IPv6 transitions.
  return { url, address: ipv4[0] };
}

export async function fetchSource(value: string, redirects = 0): Promise<{ content: Buffer; source: string; filename: string; contentType: string }> {
  if (redirects > 4) throw new LabError(400, "Too many source redirects.");
  const { url, address } = await validateDestination(value);
  const response = await new Promise<{ status: number; location?: string; type: string; content: Buffer }>((resolve, reject) => {
    // Pin the validated address for the actual connection: a second DNS lookup
    // cannot rebind the request onto a private address. TLS still verifies hostname.
    const req = httpsRequest(url, { method: "GET", headers: { "Accept-Encoding": "identity", "User-Agent": "Eidos-Sentinel-Import/1" },
      lookup: ((_host: any, options: any, callback: any) => options.all ? callback(null, [address]) : callback(null, address.address, address.family)) as any,
    }, res => {
      const status = res.statusCode || 500;
      if ([301, 302, 303, 307, 308].includes(status)) { res.resume(); resolve({ status, location: res.headers.location, type: "", content: Buffer.alloc(0) }); return; }
      if (status !== 200) { res.resume(); reject(new LabError(400, `Source returned HTTP ${status}. It may require an authenticated integration.`)); return; }
      if (res.headers["content-encoding"] && res.headers["content-encoding"] !== "identity") { res.destroy(); reject(new LabError(400, "Compressed HTTP responses are outside this public importer envelope.")); return; }
      if (Number(res.headers["content-length"]) > LIMITS.uploadBytes) { res.destroy(); reject(new LabError(413, "Source exceeds the 2 MB download limit.")); return; }
      const chunks: Buffer[] = []; let bytes = 0;
      res.on("data", chunk => { bytes += chunk.length; if (bytes > LIMITS.uploadBytes) { res.destroy(); reject(new LabError(413, "Source exceeds the 2 MB download limit.")); } else chunks.push(chunk); });
      res.on("end", () => resolve({ status, content: Buffer.concat(chunks), type: String(res.headers["content-type"] || "") }));
      res.on("error", reject);
    });
    const timer = setTimeout(() => req.destroy(new LabError(408, "Source download exceeded 20 seconds.")), 20000);
    req.on("close", () => clearTimeout(timer)); req.on("error", reject); req.end();
  });
  if (response.location) return fetchSource(new URL(response.location, url).toString(), redirects + 1);
  let filename = decodeURIComponent(url.pathname.split("/").pop() || "source").replace(/[^\w. -]/g, "_").slice(0, 120);
  if (!/\.(csv|xlsx|parquet|jsonl?|ndjson|log|pdf|html?|txt|md)$/i.test(filename)) {
    const extension = response.type.includes("html") ? ".html" : response.type.includes("json") ? ".json" : response.type.includes("csv") ? ".csv" : response.type.includes("pdf") ? ".pdf" : ".txt";
    filename += extension;
  }
  return { content: response.content, source: url.toString(), filename, contentType: response.type };
}
