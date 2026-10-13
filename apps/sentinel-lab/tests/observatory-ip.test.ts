import assert from "node:assert/strict";
import test from "node:test";
import { EventEmitter } from "node:events";
import * as https from "node:https";
import type * as http from "node:http";
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createIpObservationHandler, isPublicIp, normalizeIpTarget, observeIp } from "../lib/observatory/ip-observer";

const secret = "test-observatory-collector-secret-32-bytes";
const request = (target: unknown = "1.1.1.1", overrides: RequestInit = {}) => new Request("https://lab.test/api/observatory/ip-observe", {
  method: "POST", headers: { authorization: `Bearer ${secret}`, "content-type": "application/json" },
  body: JSON.stringify({ target }), ...overrides,
});

type RequestFactory = NonNullable<NonNullable<Parameters<typeof observeIp>[1]>["requestFactory"]>;
function fakeTransport(headers: http.IncomingHttpHeaders = {}, status = 200, inspect?: (options: https.RequestOptions) => void): RequestFactory {
  return (options, callback) => {
    inspect?.(options);
    const outgoing = new EventEmitter() as http.ClientRequest;
    outgoing.destroy = (() => outgoing) as http.ClientRequest["destroy"];
    outgoing.end = (() => {
      queueMicrotask(() => {
        const incoming = new EventEmitter() as http.IncomingMessage;
        incoming.statusCode = status;
        incoming.headers = headers;
        incoming.destroy = (() => incoming) as http.IncomingMessage["destroy"];
        incoming.read = (() => { assert.fail("Response bodies must never be read"); }) as http.IncomingMessage["read"];
        incoming.resume = (() => { assert.fail("Response bodies must never be resumed"); }) as http.IncomingMessage["resume"];
        callback(incoming);
      });
      return outgoing;
    }) as http.ClientRequest["end"];
    return outgoing;
  };
}

test("normalization is literal-only, public-only and bound to the scheme's standard port", () => {
  assert.equal(normalizeIpTarget("1.1.1.1").url, "https://1.1.1.1/");
  assert.equal(normalizeIpTarget("https://[2606:4700:4700:0:0:0:0:1111]:443/status").url, "https://[2606:4700:4700::1111]/status");
  assert.equal(normalizeIpTarget("2606:4700:4700::1111").host, "2606:4700:4700::1111");
  assert.equal(normalizeIpTarget("http://1.1.1.1:80/").port, 80);
  for (const value of ["http://1.1.1.1:443", "https://1.1.1.1:80", "https://1.1.1.1:8080", "https://user:pass@1.1.1.1", "https://1.1.1.1?token=secret", "https://1.1.1.1#", "ftp://1.1.1.1", "1.1.1.1\r\nAuthorization: secret", "https://1.1.1.1\\@127.0.0.1", " https://1.1.1.1", "example.com", "https://0x01010101", "https://16843009", "https://001.001.001.001", "https://%31.1.1.1", "https://[fe80::1%25eth0]", "https://1.1.1.1/" + "a".repeat(513)]) {
    assert.throws(() => normalizeIpTarget(value), Error, value);
  }
});

test("SSRF excludes reserved IPv4, mapped IPv6, metadata, tunnels and special-use ranges", () => {
  for (const value of ["0.0.0.0", "0.1.2.3", "10.0.0.1", "100.64.0.1", "127.0.0.1", "169.254.169.254", "172.31.255.255", "192.0.0.9", "192.0.2.1", "192.88.99.1", "192.168.0.1", "198.18.1.1", "198.51.100.1", "203.0.113.1", "224.0.0.1", "240.0.0.1", "255.255.255.255", "::", "::1", "::ffff:127.0.0.1", "::ffff:8.8.8.8", "64:ff9b::a00:1", "fc00::1", "fe80::1", "ff02::1", "2001::1", "2001:100::1", "2001:db8::1", "2002:7f00:1::", "3fff::1"]) {
    assert.equal(isPublicIp(value), false, value);
    assert.throws(() => normalizeIpTarget(value), Error, value);
  }
  for (const value of ["1.1.1.1", "8.8.8.8", "93.184.216.34", "2606:4700:4700::1111", "2001:4860:4860::8888"]) assert.equal(isPublicIp(value), true, value);
});

test("auth fails closed before parsing input or invoking the collector", async () => {
  let calls = 0;
  const collector = async (target: ReturnType<typeof normalizeIpTarget>) => { calls += 1; return observeIp(target, { requestFactory: fakeTransport() }); };
  for (const value of [undefined, "", "short", "a".repeat(31), "secret with spaces".repeat(3)]) {
    const handle = createIpObservationHandler({ getSecret: () => value, collector });
    assert.equal((await handle(request())).status, 503);
  }
  const handle = createIpObservationHandler({ getSecret: () => secret, collector });
  for (const authorization of ["", "Bearer wrong", `bearer ${secret}`, `Basic ${secret}`, `Bearer ${secret}extra`, "a".repeat(1025)]) {
    const response = await handle(request("127.0.0.1", { headers: { authorization, "content-type": "application/json" } }));
    assert.equal(response.status, 401);
    assert.ok(!(await response.text()).includes(secret));
  }
  assert.equal(calls, 0);
  assert.equal((await handle(request())).status, 200);
  assert.equal(calls, 1);
});

test("bounded payload validation never gives network transport to unauthorized or invalid input", async () => {
  let calls = 0;
  const handle = createIpObservationHandler({ getSecret: () => secret, collector: async target => { calls += 1; return observeIp(target, { requestFactory: fakeTransport() }); } });
  assert.equal((await handle(request("1.1.1.1", { body: "{" }))).status, 400);
  assert.equal((await handle(request("1.1.1.1", { body: JSON.stringify({ target: "1.1.1.1", headers: { Authorization: "secret" } }) }))).status, 400);
  assert.equal((await handle(request("1.1.1.1", { body: "x".repeat(1025) }))).status, 413);
  assert.equal((await handle(request("1.1.1.1", { headers: { authorization: `Bearer ${secret}`, "content-type": "application/json", "content-length": "1025" } }))).status, 413);
  assert.equal((await handle(request("127.0.0.1"))).status, 400);
  assert.equal((await handle(request("example.com"))).status, 400);
  assert.equal((await handle(request("1.1.1.1", { headers: { authorization: `Bearer ${secret}`, "content-type": "text/plain" } }))).status, 415);
  assert.equal(calls, 0);
});

test("one pinned HEAD sends no client credentials, follows no redirect and retains only enum metadata", async () => {
  let calls = 0;
  const result = await observeIp("https://1.1.1.1", { requestFactory: fakeTransport({
    server: "server secret-cookie=abc", "content-type": "text/html; token=secret", "set-cookie": ["session=secret"], authorization: "secret",
    location: "http://169.254.169.254/latest/meta-data/", "strict-transport-security": "max-age=31536000", "x-frame-options": "DENY", "cache-control": "private, no-store", 
  }, 302, options => {
    calls += 1;
    assert.equal(options.hostname, "1.1.1.1");
    assert.equal(options.method, "HEAD");
    assert.equal(options.port, 443);
    assert.equal(options.agent, false);
    assert.equal(options.rejectUnauthorized, true);
    assert.equal(options.servername, "");
    assert.equal(options.maxHeaderSize, 16384);
    assert.deepEqual(Object.keys(options.headers || {}).sort(), ["Accept", "Cache-Control", "Connection", "Host", "Pragma", "User-Agent"]);
    const mismatch = options.checkServerIdentity!("ignored-name", { subjectaltname: "DNS:one.one.one.one" } as never);
    assert.equal((mismatch as NodeJS.ErrnoException | undefined)?.code, "ERR_TLS_CERT_ALTNAME_INVALID");
    assert.equal(options.checkServerIdentity!("ignored-name", { subjectaltname: "IP Address:1.1.1.1" } as never), undefined);
  }) });
  assert.equal(calls, 1);
  assert.equal(result.event.status, 302);
  assert.equal(result.event.redirect?.followed, false);
  assert.equal(result.event.redirect?.target, null);
  assert.equal(result.event.redirect?.validation, "withheld_outside_boundary");
  assert.equal(result.event.response_chain_complete, false);
  assert.equal(result.event.coverage.dns.address_pinned, true);
  assert.equal(result.event.coverage.body_collected, false);
  assert.equal(result.event.headers.server, "other");
  assert.equal(result.event.headers["content-type"], "text/html");
  assert.equal(result.event.security_headers.frame_options, "deny");
  assert.equal(result.event.security_headers.cache_private, true);
  assert.equal(result.event.security_headers.cache_no_store, true);
  assert.ok(!JSON.stringify(result).includes("secret"));
});

test("transport and TLS failures are redacted; slow transport is destroyed at the total deadline", async () => {
  for (const [code, classification] of [["ECONNREFUSED", "transport_error"], ["ERR_TLS_CERT_ALTNAME_INVALID", "tls_error"], ["CERT_HAS_EXPIRED", "tls_error"]]) {
    const result = await observeIp("1.1.1.1", { requestFactory: () => { throw Object.assign(new Error("Bearer supersecret; cookie=private; https://internal.example/secret"), { code }); } });
    assert.equal(result.event.error?.code, classification);
    assert.equal(result.event.status, null);
    assert.ok(!JSON.stringify(result).includes("supersecret"));
    assert.ok(!JSON.stringify(result).includes("internal.example"));
  }
  let destroyed = false;
  const result = await observeIp("1.1.1.1", { timeoutMs: 20, requestFactory: () => {
    const outgoing = new EventEmitter() as http.ClientRequest;
    outgoing.end = (() => outgoing) as http.ClientRequest["end"];
    outgoing.destroy = (() => { destroyed = true; return outgoing; }) as http.ClientRequest["destroy"];
    return outgoing;
  } });
  assert.equal(result.event.error?.code, "timeout");
  assert.equal(result.event.status, null);
  assert.equal(destroyed, true);
});

test("actual Node TLS rejects a self-signed server without an insecure retry", async () => {
  const directory = await mkdtemp(join(tmpdir(), "observatory-tls-"));
  let server: https.Server | undefined;
  let requests = 0;
  try {
    execFileSync("openssl", ["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=1.1.1.1", "-addext", "subjectAltName=IP:1.1.1.1", "-keyout", join(directory, "key.pem"), "-out", join(directory, "cert.pem")], { stdio: "ignore" });
    server = https.createServer({ key: await readFile(join(directory, "key.pem")), cert: await readFile(join(directory, "cert.pem")) }, (_request, response) => { requests += 1; response.writeHead(200); response.end(); });
    await new Promise<void>(resolve => server!.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    assert.ok(address && typeof address !== "string");
    let attempts = 0;
    const result = await observeIp("1.1.1.1", { requestFactory: (options, callback) => {
      attempts += 1;
      // Test-only local fixture transport. Production input can never select
      // this injection or override the validated connection destination.
      return https.request({ ...options, hostname: "127.0.0.1", port: address.port }, callback);
    } });
    assert.equal(result.event.error?.code, "tls_error");
    assert.equal(result.event.status, null);
    assert.equal(attempts, 1);
    assert.equal(requests, 0);
  } finally {
    if (server) await new Promise<void>((resolve, reject) => server!.close(error => error ? reject(error) : resolve()));
    await rm(directory, { recursive: true, force: true });
  }
});

test("rate limit applies to the IP across scheme and path variations, then releases at 15 seconds", async () => {
  let now = 100;
  let calls = 0;
  const handle = createIpObservationHandler({ getSecret: () => secret, clock: () => now, collector: async target => { calls += 1; return observeIp(target, { requestFactory: fakeTransport() }); } });
  assert.equal((await handle(request("https://1.1.1.1/a"))).status, 200);
  const duplicate = await handle(request("http://1.1.1.1/b"));
  assert.equal(duplicate.status, 429);
  assert.equal(duplicate.headers.get("retry-after"), "15");
  now += 14_999;
  assert.equal((await handle(request("1.1.1.1"))).status, 429);
  now += 1;
  assert.equal((await handle(request("1.1.1.1"))).status, 200);
  assert.equal(calls, 2);
});

test("only four observations can be active and slots are released after completion", async () => {
  const releases: Array<() => void> = [];
  const handle = createIpObservationHandler({ getSecret: () => secret, collector: target => new Promise(resolve => {
    releases.push(async () => resolve(await observeIp(target, { requestFactory: fakeTransport() })));
  }) });
  const pending = [1, 2, 3, 4].map(last => handle(request(`8.8.8.${last}`)));
  while (releases.length !== 4) await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal((await handle(request("9.9.9.9"))).status, 429);
  releases.forEach(release => release());
  const responses = await Promise.all(pending);
  assert.ok(responses.every(response => response.status === 200));
  const next = handle(request("9.9.9.9"));
  while (Number(releases.length) !== 5) await new Promise<void>(resolve => setImmediate(resolve));
  releases[4]();
  assert.equal((await next).status, 200);
});

test("rate key capacity fails closed rather than evicting unexpired IP cooldowns", async () => {
  const handle = createIpObservationHandler({ getSecret: () => secret, clock: () => 0, collector: async target => observeIp(target, { requestFactory: fakeTransport() }) });
  for (let index = 0; index < 512; index += 1) assert.equal((await handle(request(`8.10.${Math.floor(index / 256)}.${index % 256}`))).status, 200);
  assert.equal((await handle(request("9.9.9.9"))).status, 429);
  assert.equal((await handle(request("8.10.0.0"))).status, 429);
});
