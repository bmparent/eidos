import test from "node:test";
import assert from "node:assert/strict";
import { collectObservation, collectIpRelayObservation, handleObservation, ipCollectorConfiguration, normalizeTarget } from "./collector.mjs";

const publicA = "93.184.215.14";
const publicAAAA = "2606:4700:4700::1111";

function fixtureFetch({ a = publicA, aaaa = publicAAAA, status = 200, headers = {}, dnsResponse, headError, body = "private body must never be read" } = {}) {
  const calls = [];
  const fetchImpl = async (url, options) => {
    calls.push({ url, options });
    if (url.startsWith("https://cloudflare-dns.com/dns-query?")) {
      const type = new URL(url).searchParams.get("type");
      const address = type === "A" ? a : aaaa;
      const payload = dnsResponse || { Status: 0, Answer: address ? [{ type: type === "A" ? 1 : 28, data: address }] : [] };
      return new Response(JSON.stringify(payload), { headers: { "content-type": "application/dns-json" } });
    }
    if (headError) throw headError;
    const response = new Response(body, { status, headers });
    for (const method of ["text", "json", "arrayBuffer", "blob"]) response[method] = () => { throw new Error("Target body must not be read"); };
    return response;
  };
  return { fetchImpl, calls };
}

function request(target, { origin = "https://observatory.example.com", contentType = "application/json", extra = {}, method = "POST", raw, path = "/api/observe" } = {}) {
  const headers = { "Content-Type": contentType, ...extra };
  if (origin !== null) headers.Origin = origin;
  return new Request(`https://observatory.example.com${path}`, { method, headers, body: method === "POST" ? raw ?? JSON.stringify({ target }) : undefined });
}

async function mockedGlobalFetch(fetchImpl, fn) {
  const original = globalThis.fetch;
  globalThis.fetch = fetchImpl;
  try { return await fn(); } finally { globalThis.fetch = original; }
}

test("normalizes domain, IDN, explicit standard ports, IPv4 and IPv6 without network", () => {
  assert.equal(normalizeTarget("EIDOS-WORKS.COM/account/").url, "https://eidos-works.com/account/");
  assert.equal(normalizeTarget("https://eidos-works.com.:443/").url, "https://eidos-works.com/");
  assert.equal(normalizeTarget("http://example.com:80/").url, "http://example.com/");
  assert.equal(normalizeTarget("https://bücher.de/").hostname, "xn--bcher-kva.de");
  assert.equal(normalizeTarget("8.8.8.8").kind, "ipv4");
  assert.equal(normalizeTarget("2606:4700:4700::1111").kind, "ipv6");
  assert.equal(normalizeTarget("https://[2606:4700:4700::1111]/").kind, "ipv6");
  assert.equal(normalizeTarget("https://eidos-works.com").id, normalizeTarget("eidos-works.com/").id);
});

test("rejects SSRF special addresses including alternative IPv4 encodings and IPv6 tunnels", () => {
  const rejected = [
    "localhost", "localhost.", "a.localhost", "a.internal", "metadata.google.internal", "host.local", "router.lan", "home.arpa", "test.invalid", "internal", "a.onion",
    "127.0.0.1", "127.1", "2130706433", "0x7f000001", "0177.0.0.1", "0x7f.0.0.1", "https://%31%32%37.0.0.1/",
    "0.0.0.0", "10.1.1.1", "100.64.0.1", "169.254.169.254", "172.31.1.1", "192.168.1.1", "192.0.0.9", "192.0.2.1", "198.19.1.1", "198.51.100.1", "203.0.113.2", "224.0.0.1", "255.255.255.255",
    "[::1]", "[::]", "[::ffff:127.0.0.1]", "[::ffff:8.8.8.8]", "[fe80::1]", "[fc00::1]", "[ff02::1]", "[64:ff9b::808:808]", "[2001:db8::1]", "[2001::1]", "[2002:0808:0808::1]", "[3fff::1]",
  ];
  for (const value of rejected) assert.throws(() => normalizeTarget(value), undefined, value);
});

test("rejects credentials, control input, nonstandard ports, schemes, query, fragments and recursion", () => {
  const rejected = ["https://user:pass@example.com/", "https://user@example.com/", "ftp://example.com/", "file:///etc/passwd", "https://example.com:22/", "https://example.com:8080/", "https://example.com/?", "https://example.com/?token=secret", "https://example.com/#", "https://example.com/#secret", "https://exam\nple.com/", "https://example.com\\@127.0.0.1/", "https://example.com/a b"];
  for (const value of rejected) assert.throws(() => normalizeTarget(value), undefined, value);
  assert.throws(() => normalizeTarget("https://OBSERVATORY.example.com./path", { blockedHosts: ["observatory.example.com"] }), (e) => e.code === "recursive_target");
});

test("exactly one HEAD after public DNS validation, no forwarded secrets or body reads", async () => {
  const mock = fixtureFetch({ headers: { "set-cookie": "private-cookie", authorization: "private-auth", server: "x".repeat(1000), "cf-cache-status": "DYNAMIC", "content-type": "text/html; charset=utf-8", "strict-transport-security": "max-age=1000", "content-security-policy": "secret-policy", "x-content-type-options": "nosniff", "x-frame-options": "DENY", "cache-control": "private, no-store" } });
  const result = await collectObservation("https://eidos-works.com/", { fetchImpl: mock.fetchImpl });
  assert.equal(mock.calls.length, 3);
  assert.equal(mock.calls.filter((call) => call.options.method === "HEAD").length, 1);
  const head = mock.calls.at(-1);
  assert.equal(head.options.redirect, "manual");
  assert.equal(Object.hasOwn(head.options, "cache"), false);
  assert.equal(head.options.headers["Cache-Control"], "no-cache");
  assert.equal(head.options.headers.Pragma, "no-cache");
  assert.equal(head.options.cf.cacheTtlByStatus["100-599"], -1);
  assert.deepEqual(Object.keys(head.options.headers).sort(), ["Accept", "Cache-Control", "Pragma", "User-Agent"]);
  assert.equal(result.event.status, 200);
  assert.equal(result.event.headers.server.length, 256);
  assert.equal(result.event.coverage.head_attempted, true);
  assert.equal(result.event.coverage.dns.address_pinned, false);
  assert.equal(result.event.security_headers.csp_present, true);
  assert.equal(result.event.security_headers.frame_options, "deny");
  assert.equal(result.event.security_headers.cache_private, true);
  assert.equal(result.event.security_headers.cache_no_store, true);
  assert.equal(result.retry_after_ms, 15000);
  assert.ok(result.event.latency_ms >= 0);
  assert.doesNotMatch(JSON.stringify(result), /private-cookie|private-auth|secret-policy|private body/);
});

test("all returned DNS addresses must be public; mixed records fail before HEAD", async () => {
  for (const options of [ { a: "127.0.0.1" }, { aaaa: "::ffff:127.0.0.1" }, { a: "169.254.169.254" }, { dnsResponse: { Status: 0, Answer: [{ type: 1, data: publicA }, { type: 28, data: "fd00::1" }] } } ]) {
    const mock = fixtureFetch(options);
    const result = await collectObservation("example.com", { fetchImpl: mock.fetchImpl });
    assert.equal(result.event.error.code, "dns_non_public");
    assert.equal(result.event.status, null);
    assert.equal(result.event.latency_ms, null);
    assert.equal(result.event.coverage.head_attempted, false);
    assert.ok(mock.calls.every((call) => call.options.method === "GET"));
    assert.ok(Object.values(result.event.security_headers).every((value) => value === null));
  }
});

test("DNS empty/failed/oversized/malicious aliases fail closed", async () => {
  const fixtures = [
    { Status: 0, Answer: [] }, { Status: 2 }, { Status: 3 }, { Status: 0, TC: true },
    { Status: 0, Answer: [{ type: 5, data: "metadata.google.internal." }, { type: 1, data: publicA }] },
    { Status: 0, Answer: [{ type: 5, data: "observatory.example.com." }, { type: 1, data: publicA }] },
    { Status: 0, Answer: Array.from({ length: 33 }, () => ({ type: 1, data: publicA })) },
  ];
  for (const dnsResponse of fixtures) {
    const mock = fixtureFetch({ dnsResponse });
    const result = await collectObservation("example.com", { fetchImpl: mock.fetchImpl, blockedHosts: ["observatory.example.com"] });
    assert.equal(result.event.status, null);
    assert.equal(result.event.coverage.head_attempted, false);
    assert.ok(result.event.error);
    assert.equal(mock.calls.filter((call) => call.options.method === "HEAD").length, 0);
  }
  const oversize = await collectObservation("example.com", { fetchImpl: async () => new Response("a".repeat(16385)) });
  assert.equal(oversize.event.error.code, "dns_invalid_response");
});

test("redirects are recorded once, never followed, unsafe destinations withheld", async () => {
  for (const [location, expected] of [["/next", "https://eidos-works.com/next"], ["https://other-public.com/next", "https://other-public.com/next"], ["http://127.0.0.1/", null], ["https://example.com/?secret=x", null], ["https://user:password@example.com/", null], ["https://observatory.example.com/", null]]) {
    const mock = fixtureFetch({ status: 302, headers: { location } });
    const result = await collectObservation("eidos-works.com", { fetchImpl: mock.fetchImpl, blockedHosts: ["observatory.example.com"] });
    assert.equal(result.event.status, 302);
    assert.equal(result.event.redirect.followed, false);
    assert.equal(result.event.redirect.target, expected);
    assert.equal(mock.calls.filter((call) => call.options.method === "HEAD").length, 1);
    assert.doesNotMatch(JSON.stringify(result), /secret=x|password/);
  }
});

test("transport failures are unknown status without raw exceptions or site outage claims", async () => {
  const mock = fixtureFetch({ headError: new Error("secret token localhost private proxy") });
  const result = await collectObservation("example.com", { fetchImpl: mock.fetchImpl });
  assert.equal(result.event.status, null);
  assert.equal(result.event.error.code, "transport_error");
  assert.equal(result.event.error.scope, "observation_vantage");
  assert.equal(result.event.coverage.head_attempted, true);
  assert.match(result.event.error.message, /availability is unknown/);
  assert.doesNotMatch(JSON.stringify(result), /secret token|private proxy/);
});

test("body cancellation cleanup failure does not erase already observed response", async () => {
  const mock = fixtureFetch();
  const fetchImpl = async (url, options) => {
    if (options.method !== "HEAD") return mock.fetchImpl(url, options);
    return { status: 200, redirected: false, headers: new Headers({ server: "test" }), body: { cancel: async () => { throw new Error("cleanup only"); } } };
  };
  const result = await collectObservation("example.com", { fetchImpl });
  assert.equal(result.event.status, 200);
  assert.equal(result.event.error, null);
  assert.equal(result.event.response_chain_complete, true);
});

test("pre-cancelled observation has no outbound requests", async () => {
  const controller = new AbortController(); controller.abort();
  let calls = 0;
  const result = await collectObservation("example.com", { signal: controller.signal, fetchImpl: () => { calls++; throw new Error("Should not run"); } });
  assert.equal(calls, 0);
  assert.equal(result.event.error.code, "cancelled");
});

test("IP limitation is an actionable 422 error, never a synthetic observation", async () => {
  await mockedGlobalFetch(() => { throw new Error("No network allowed"); }, async () => {
    const response = await handleObservation(request("8.8.8.8"));
    const body = await response.json();
    assert.equal(response.status, 422);
    assert.equal(body.error.code, "platform_ip_unsupported");
    assert.match(body.error.message, /local_server.py/);
    assert.equal(body.event, undefined);
    await assert.rejects(collectObservation("8.8.8.8"), (error) => error.code === "platform_ip_unsupported");
  });
});

test("HTTP API enforces same origin, POST, exact small JSON and recursion denial", async () => {
  await mockedGlobalFetch(() => { throw new Error("No network allowed"); }, async () => {
    for (const [req, expected] of [
      [request("example.com", { method: "GET" }), 405],
      [request("example.com", { origin: null }), 403],
      [request("example.com", { origin: "https://evil.example" }), 403],
      [request("example.com", { extra: { "Sec-Fetch-Site": "cross-site" } }), 403],
      [request("example.com", { contentType: "text/plain" }), 415],
      [request("example.com", { raw: '{"target":"example.com","method":"GET"}' }), 400],
      [request("example.com", { raw: "badjson" }), 400],
      [request("example.com", { raw: "x".repeat(1025) }), 400],
      [request("https://observatory.example.com/"), 400],
    ]) {
      const response = await handleObservation(req);
      assert.equal(response.status, expected);
      assert.ok((await response.json()).error.code);
    }
  });
});

test("canonical target rate limit reserves before work and forwards no client credentials", async () => {
  const mock = fixtureFetch();
  await mockedGlobalFetch(mock.fetchImpl, async () => {
    const first = await handleObservation(request("https://RATE.example.com:443/", { extra: { Cookie: "private-user-cookie", Authorization: "Bearer private-api-token", "X-Forwarded-For": "private-client-IP" } }));
    assert.equal(first.status, 200);
    const second = await handleObservation(request("rate.example.com/"));
    assert.equal(second.status, 429);
    const body = await second.json();
    assert.ok(body.retry_after_ms > 0 && body.retry_after_ms <= 15000);
    assert.match(body.rate_limit_scope, /not durable/);
    assert.equal(mock.calls.filter((call) => call.options.method === "HEAD").length, 1);
    assert.doesNotMatch(JSON.stringify(mock.calls), /private-user-cookie|private-api-token|private-client-IP/);
  });
});

test("default global fetch retains the runtime receiver for DNS and HEAD", async () => {
  const mock = fixtureFetch();
  const receivers = [];
  await mockedGlobalFetch(async function receiverSensitiveFetch(url, options) {
    receivers.push(this);
    if (this !== globalThis) throw new TypeError("Illegal invocation: function called with incorrect this reference");
    return mock.fetchImpl(url, options);
  }, async () => {
    const result = await collectObservation("eidos-works.com");
    assert.equal(result.event.status, 200);
    assert.equal(result.event.error, null);
    assert.equal(receivers.length, 3);
    assert.ok(receivers.every((receiver) => receiver === globalThis));
  });
});

test("runtime failure codes and diagnostic logs are bounded and contain no raw messages", async () => {
  const originalError = console.error;
  const logs = [];
  console.error = (...values) => logs.push(values);
  try {
    for (const [message, classifier] of [["Illegal invocation: function called with incorrect this reference secret-token https://private.example/path", "illegal_invocation"], ["Unsupported cache mode: secret-cache-value", "unsupported_cache_mode"], ["The 'cache' field on 'RequestInitializerDict' is not implemented.", "cache_option_unimplemented"]]) {
      const result = await collectObservation("example.com", { fetchImpl: () => { throw new TypeError(message); } });
      assert.equal(result.event.error.code, "collector_runtime_error");
      assert.equal(result.event.error.classifier, classifier);
      assert.equal(result.event.coverage.failure_stage, "dns_preflight");
      assert.equal(result.event.coverage.head_attempted, false);
      assert.equal(result.event.status, null);
      assert.doesNotMatch(JSON.stringify(result), /secret-token|private\.example|secret-cache-value/);
    }
    assert.equal(logs.length, 3);
    assert.ok(logs.every((row) => row[0] === "sentinel_collector_failure"));
    assert.doesNotMatch(JSON.stringify(logs), /secret-token|private\.example|secret-cache-value/);
    assert.deepEqual(Object.keys(JSON.parse(logs[0][1])).sort(), ["classifier", "exception_name", "message", "stack_functions", "stage"]);
  } finally { console.error = originalError; }
});

test("server-only diagnostics redact credentials, URLs, addresses and opaque strings", async () => {
  const originalError = console.error;
  const logs = [];
  console.error = (...values) => logs.push(values);
  try {
    const secret = "ZXhhbXBsZXNlY3JldDEyMzQ1Njc4OTA=";
    const message = `Runtime fetch failed https://private.example/path 192.168.1.2:443 [2001:db8::1] Bearer abc123 authorization: XYZ; password=hidden; ${secret} sk-supersecret 'short-secret' ` + "x".repeat(220);
    const result = await collectObservation("example.com", { fetchImpl: () => { throw new TypeError(message); } });
    const apiText = JSON.stringify(result);
    assert.doesNotMatch(apiText, /Runtime fetch failed|abc123|XYZ|hidden|short-secret/);
    const log = JSON.parse(logs[0][1]);
    assert.ok(log.message.length <= 200);
    assert.match(log.message, /Runtime fetch failed/);
    assert.doesNotMatch(log.message, /private\.example|192\.168|2001:db8|abc123|XYZ|hidden|ZXhh|sk-supersecret|short-secret/);
    assert.ok(log.stack_functions.every((name) => ["validateDns", "collectObservation", "handleObservation", "boundedText", "fetch", "Function.fetch", "Object.fetch"].includes(name)));
    logs.length = 0;
    await collectObservation("example.com", { fetchImpl: () => { throw new TypeError("The 'cache' field on 'RequestInitializerDict' is not implemented."); } });
    assert.equal(JSON.parse(logs[0][1]).message, "The 'cache' field on 'RequestInitializerDict' is not implemented.");
  } finally { console.error = originalError; }
});

test("works with an older Worker runtime that rejects RequestInit.cache before network I/O", async () => {
  const mock = fixtureFetch();
  const result = await collectObservation("eidos-works.com", { fetchImpl: async (url, options) => {
    if (Object.hasOwn(options, "cache")) throw new TypeError("The 'cache' field on 'RequestInitializerDict' is not implemented.");
    assert.equal(options.headers["Cache-Control"], "no-cache");
    assert.equal(options.headers.Pragma, "no-cache");
    assert.equal(options.cf.cacheTtlByStatus["100-599"], -1);
    return mock.fetchImpl(url, options);
  } });
  assert.equal(mock.calls.length, 3);
  assert.equal(result.event.status, 200);
  assert.equal(result.event.error, null);
});

test("works with the deployed older edge runtime that rejects redirect error mode", async () => {
  const mock = fixtureFetch();
  const result = await collectObservation("eidos-works.com", { fetchImpl: async (url, options) => {
    if (options.redirect === "error") throw new TypeError('Invalid redirect value, must be one of "follow" or "manual" ("error" will not be implemented at the edge)');
    assert.equal(options.redirect, "manual");
    return mock.fetchImpl(url, options);
  } });
  assert.equal(mock.calls.length, 3);
  assert.equal(result.event.status, 200);
  assert.equal(result.event.error, null);
});

test("DNS redirects are rejected in manual mode without following or sending target HEAD", async () => {
  const calls = [];
  const result = await collectObservation("eidos-works.com", { fetchImpl: async (url, options) => {
    calls.push({ url, options });
    assert.equal(options.method, "GET");
    assert.equal(options.redirect, "manual");
    assert.match(url, /^https:\/\/cloudflare-dns\.com\/dns-query\?/);
    return new Response("must not be read", { status: 302, headers: { Location: "http://127.0.0.1/private" } });
  } });
  assert.equal(calls.length, 2);
  assert.equal(result.event.error.code, "dns_unavailable");
  assert.equal(result.event.status, null);
  assert.equal(result.event.coverage.head_attempted, false);
  assert.doesNotMatch(JSON.stringify(result), /127\.0\.0\.1|\/private/);
});

const relayConfig = { url: 'https://sentinel-relay.vercel.app/api/observatory/ip-observe', secret: 'a'.repeat(64), bypass: 'b'.repeat(32) };
function relayEvidence(input = 'https://1.1.1.1/') {
  const target = normalizeTarget(input);
  return {
    target: { id: target.id, url: target.url, label: target.label }, retry_after_ms: 15000,
    event: { id: 'evt_' + '1'.repeat(32), source_id: target.id, target: target.label, url: target.url,
      observed_at: '2026-10-13T20:00:00.000Z', completed_at: '2026-10-13T20:00:00.100Z',
      latency_ms: 100, elapsed_ms: 100, status: 301, kind: 'measurement', method: 'HEAD', error: null,
      headers: { server: 'cloudflare', 'set-cookie': 'private-cookie', authorization: 'private-auth' },
      security_headers: { hsts_present: true }, redirect: { followed: false, target: 'https://one.one.one.one/', validation: 'not_requested' },
      coverage: { head_attempted: true, body_collected: false, redirect_followed: false, dns: { address_pinned: true } },
      private_body: 'private body must never pass through',
    },
  };
}

test('1.1.1.1 fallback identifies named service and never claims a direct pinned IP', async () => {
  const mock = fixtureFetch();
  const result = await collectObservation('https://1.1.1.1/', { fetchImpl: mock.fetchImpl });
  assert.equal(mock.calls.length, 3);
  assert.equal(mock.calls.at(-1).url, 'https://one.one.one.one/');
  assert.ok(mock.calls.slice(0, 2).every(call => new URL(call.url).searchParams.get('name') === 'one.one.one.one'));
  assert.equal(result.target.url, 'https://1.1.1.1/');
  assert.equal(result.event.url, 'https://one.one.one.one/');
  assert.equal(result.event.source_id, result.target.id);
  assert.equal(result.event.target, '1.1.1.1 via one.one.one.one');
  assert.equal(result.event.coverage.transport.kind, 'named_service_adapter');
  assert.equal(result.event.coverage.transport.direct_ip, false);
  assert.equal(result.event.coverage.transport.address_pinned, false);
  assert.equal(result.event.coverage.transport.tls_server_name, 'one.one.one.one');
  assert.match(result.event.coverage.transport.limitation, /does not establish a connection to the literal/);
});

test('relay configuration is server-only HTTPS Vercel exact route with bounded secrets', () => {
  const env = { OBSERVATORY_COLLECTOR_URL: relayConfig.url, OBSERVATORY_COLLECTOR_SECRET: relayConfig.secret, OBSERVATORY_COLLECTOR_BYPASS_TOKEN: relayConfig.bypass };
  assert.deepEqual(ipCollectorConfiguration(env), relayConfig);
  for (const value of ['http://sentinel.vercel.app/api/observatory/ip-observe', 'https://attacker.example/api/observatory/ip-observe', 'https://sentinel.vercel.app/other', 'https://sentinel.vercel.app/api/observatory/ip-observe?token=secret', 'https://user:pass@sentinel.vercel.app/api/observatory/ip-observe']) assert.equal(ipCollectorConfiguration({ ...env, OBSERVATORY_COLLECTOR_URL: value }), null);
  assert.equal(ipCollectorConfiguration({ ...env, OBSERVATORY_COLLECTOR_SECRET: '' }), null);
  assert.equal(ipCollectorConfiguration({ ...env, OBSERVATORY_COLLECTOR_BYPASS_TOKEN: 'unsafe\r\nheader' }), null);
});

test('pinned IP relay sends one server POST and accepts only matching bounded evidence', async () => {
  const calls = [];
  const result = await collectIpRelayObservation('1.1.1.1', relayConfig, { fetchImpl: async (url, options) => {
    calls.push({ url, options }); return Response.json(relayEvidence());
  } });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, relayConfig.url);
  assert.equal(calls[0].options.method, 'POST');
  assert.equal(calls[0].options.redirect, 'manual');
  assert.equal(calls[0].options.headers.Authorization, 'Bearer ' + relayConfig.secret);
  assert.equal(calls[0].options.headers['x-vercel-protection-bypass'], relayConfig.bypass);
  assert.deepEqual(JSON.parse(calls[0].options.body), { target: 'https://1.1.1.1/' });
  assert.equal(result.event.status, 301);
  assert.equal(result.event.url, 'https://1.1.1.1/');
  assert.equal(result.event.coverage.transport.kind, 'authenticated_pinned_ip_relay');
  assert.equal(result.event.coverage.transport.direct_ip, true);
  assert.equal(result.event.coverage.transport.address_pinned, true);
  assert.equal(result.event.coverage.dns.address_pinned, true);
  assert.equal(result.event.redirect.followed, false);
  assert.doesNotMatch(JSON.stringify(result), /private-cookie|private-auth|private body|aaaaaa|bbbbbb/);
});

test('private IPs and DNS names never dispatch to the direct IP relay', async () => {
  let calls = 0;
  for (const value of ['127.0.0.1', '10.0.0.1', '169.254.169.254', '[::1]', '[::ffff:1.1.1.1]', 'public.example.com']) await assert.rejects(collectIpRelayObservation(value, relayConfig, { fetchImpl: () => { calls++; throw Error('must not dispatch'); } }));
  assert.equal(calls, 0);
});

test('mismatched targets, unpinned transport, body capture and inconsistent times fail relay evidence validation', async () => {
  const mutations = [
    p => { p.event.url = 'https://other-public.com/'; },
    p => { p.event.source_id = 'other_source'; },
    p => { p.event.coverage.dns.address_pinned = false; },
    p => { p.event.coverage.body_collected = true; },
    p => { p.event.coverage.redirect_followed = true; },
    p => { p.event.completed_at = '2026-10-13T19:00:00Z'; },
    p => { p.event.error = { code: 'transport_error' }; },
    p => { p.event.status = null; },
  ];
  for (const mutate of mutations) {
    const payload = relayEvidence(); mutate(payload);
    await assert.rejects(collectIpRelayObservation('1.1.1.1', relayConfig, { fetchImpl: async () => Response.json(payload) }), e => e.code === 'collector_invalid_response');
  }
  await assert.rejects(collectIpRelayObservation('1.1.1.1', relayConfig, { fetchImpl: async () => new Response('x'.repeat(32769)) }), e => e.code === 'collector_invalid_response');
});

test('IP relay refuses redirects/protection errors and never forwards arbitrary remote messages', async () => {
  for (const [status, code] of [[302, 'collector_unavailable'], [401, 'collector_auth_failed'], [403, 'collector_auth_failed'], [429, 'sample_rate_limited'], [500, 'collector_unavailable']]) {
    await assert.rejects(collectIpRelayObservation('1.1.1.1', relayConfig, { fetchImpl: async () => new Response('private proxy password', { status, headers: { Location: 'http://127.0.0.1/' } }) }), e => e.code === code && !/private proxy/.test(e.message));
  }
  const payload = relayEvidence(); payload.event.status = null; payload.event.error = { code: 'tls_error', message: 'private certificate details' };
  const result = await collectIpRelayObservation('1.1.1.1', relayConfig, { fetchImpl: async () => Response.json(payload) });
  assert.equal(result.event.status, null);
  assert.equal(result.event.error.code, 'tls_error');
  assert.match(result.event.error.message, /no insecure retry/);
  assert.doesNotMatch(JSON.stringify(result), /private certificate/);
});

test('hosted API generic public IP routes to configured collector without visitor credentials', async () => {
  const calls = [];
  const env = { OBSERVATORY_COLLECTOR_URL: relayConfig.url, OBSERVATORY_COLLECTOR_SECRET: relayConfig.secret, OBSERVATORY_COLLECTOR_BYPASS_TOKEN: relayConfig.bypass };
  await mockedGlobalFetch(async (url, options) => { calls.push({ url, options }); return Response.json(relayEvidence('https://9.9.9.9/')); }, async () => {
    const response = await handleObservation(request('9.9.9.9', { extra: { Cookie: 'visitor-cookie', Authorization: 'Bearer visitor-token', 'X-Forwarded-For': 'visitor-address' } }), env);
    assert.equal(response.status, 200);
    const result = await response.json();
    assert.equal(result.event.url, 'https://9.9.9.9/');
    assert.equal(result.event.coverage.transport.direct_ip, true);
    assert.equal(calls.length, 1);
    assert.doesNotMatch(JSON.stringify(calls), /visitor-cookie|visitor-token|visitor-address/);
  });
});

test('cancelled IP relay has no outbound request and preserves a clear cancellation result', async () => {
  const controller = new AbortController(); controller.abort();
  let calls = 0;
  await assert.rejects(collectIpRelayObservation('1.1.1.1', relayConfig, { signal: controller.signal, fetchImpl: () => { calls++; throw Error('must not run'); } }), e => e.code === 'collector_cancelled');
  assert.equal(calls, 0);
});

test('IP socket metadata is allowlisted and public-only, unavailable metrics stay null', async () => {
  const payload = relayEvidence();
  payload.event.coverage.network = {peer_ip:'1.1.1.1',peer_port:443,socket_connected:true,tls_verified:true,tls_protocol:'TLSv1.3',bytes_written:0,bytes_read:380,local_ip:'10.0.0.9',secret:'private-socket-secret'};
  const accepted = await collectIpRelayObservation('1.1.1.1', relayConfig, {fetchImpl:async()=>Response.json(payload)});
  assert.equal(accepted.event.coverage.network.peer_matches_destination,true);
  assert.equal(accepted.event.coverage.network.tls_verified,true);
  assert.equal(accepted.event.coverage.network.bytes_written,0);
  assert.equal(accepted.event.coverage.network.bytes_read,380);
  assert.doesNotMatch(JSON.stringify(accepted),/private-socket-secret|10\.0\.0\.9/);
  payload.event.coverage.network = {peer_ip:'10.0.0.8',peer_port:443,tls_protocol:'Bearer secret',bytes_written:-1,bytes_read:'not observed'};
  const withheld = await collectIpRelayObservation('1.1.1.1', relayConfig, {fetchImpl:async()=>Response.json(payload)});
  assert.equal(withheld.event.coverage.network.peer_ip,null);
  assert.equal(withheld.event.coverage.network.tls_verified,null);
  assert.equal(withheld.event.coverage.network.bytes_written,null);
  assert.equal(withheld.event.coverage.network.bytes_read,null);
  assert.doesNotMatch(JSON.stringify(withheld),/10\.0\.0\.8|Bearer secret/);
});

test('public IP collector cooldown includes retry interval on hosted API', async () => {
  const env = { OBSERVATORY_COLLECTOR_URL: relayConfig.url, OBSERVATORY_COLLECTOR_SECRET: relayConfig.secret };
  await mockedGlobalFetch(async () => new Response('not evidence', { status: 429 }), async () => {
    const response = await handleObservation(request('https://9.9.9.10/'), env);
    assert.equal(response.status, 429);
    assert.equal(response.headers.get('Retry-After'), '15');
    const result = await response.json();
    assert.equal(result.error.code, 'sample_rate_limited');
    assert.equal(result.retry_after_ms, 15000);
    assert.equal(result.event, undefined);
  });
});
