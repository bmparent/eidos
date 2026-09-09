import { randomUUID } from "node:crypto";
import { guidedStore, hash, LabError, LIMITS, type Document } from "./store";
import { principal, requireSameOrigin, signIn } from "./auth";
import { cancelJob, pollJob, publicJob, startJob } from "./jobs";
import { fetchSource } from "./fetch-source";
import { createMonitor, ingest, issueKey, monitorStatus, processMonitor, revokeKeys } from "./telemetry";

const headers = { "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff" };
const json = (value: unknown, status = 200) => Response.json(value, { status, headers });
async function body(request: Request, maximum = 2_800_000): Promise<Document> {
  if (Number(request.headers.get("Content-Length")) > maximum) throw new LabError(413, "Request exceeds the upload envelope.");
  const reader = request.body?.getReader();
  if (!reader) throw new LabError(400, "A JSON request body is required.");
  let bytes = 0; const chunks: Uint8Array[] = [];
  for (;;) {
    const result = await reader.read(); if (result.done) break;
    bytes += result.value.byteLength;
    if (bytes > maximum) { await reader.cancel(); throw new LabError(413, "Request exceeds the upload envelope."); }
    chunks.push(result.value);
  }
  try {
    const value = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (!value || Array.isArray(value) || typeof value !== "object") throw Error();
    return value;
  } catch { throw new LabError(400, "Malformed JSON request."); }
}

export async function handleGuided(request: Request, path: string[]) {
  const diagnosticId = randomUUID();
  try {
    const [resource, id, action] = path;
    const method = request.method;
    if (method !== "GET") requireSameOrigin(request);
    const store = guidedStore();
    if (resource === "worker" && id && method === "POST") {
      await store.initialize();
      const row = await store.client.execute({ sql: "SELECT owner,provider FROM sentinel_guided_jobs WHERE scope=? AND id=?", args: [store.scope, id] });
      const provider = row.rows[0]?.provider ? JSON.parse(String(row.rows[0].provider)) : null;
      const supplied = request.headers.get("Authorization")?.match(/^Bearer (\S+)$/)?.[1] || "";
      if (!provider || !supplied || hash(supplied) !== provider.callbackHash || Date.now() - provider.allocatedAt > (LIMITS.jobSeconds + 360) * 1000)
        throw new LabError(401, "Worker callback not authorized.");
      return json(publicJob(await pollJob(String(row.rows[0].owner), id, store)));
    }
    if (resource === "session" && method === "POST") {
      const session = await signIn(request, store);
      return Response.json({ user: session.user, limits: LIMITS }, { headers: { ...headers, ...(session.cookie ? { "Set-Cookie": session.cookie } : {}) } });
    }
    if (resource === "session" && method === "DELETE") {
      const local = process.env.EIDOS_GUIDED_LOCAL === "1" && !process.env.VERCEL;
      return Response.json({ signedOut: true }, { headers: { ...headers, "Set-Cookie": `${local ? "eidos_lab_local" : "__Host-eidos_lab"}=; HttpOnly; Path=/; SameSite=Strict; Max-Age=0${local ? "" : "; Secure"}` } });
    }
    const user = await principal(request, store, resource === "telemetry");
    if (resource === "session") return json({ user, limits: LIMITS });
    if (resource === "telemetry" && id && method === "POST") {
      if (user.credential !== "connector" || user.source !== id) throw new LabError(403, "A credential scoped to this telemetry source is required.");
      const receipt = await ingest(user.id, id, await body(request, 500000), store);
      let job: unknown = null, processingNotice = null;
      if (receipt.autoProcess && receipt.buffered >= 24) {
        try { job = publicJob(await processMonitor(user.id, id, store)); }
        catch (error) { processingNotice = error instanceof Error ? error.message : "Processing paused."; }
      }
      // OTLP success uses the standard empty JSON response. Custom receipts are
      // available in the source health UI; failure responses keep retry semantics.
      if (action === "v1") return json({});
      return json({ ...receipt, job, processingNotice });
    }
    if (user.credential === "connector") throw new LabError(403, "Ingest-only credential.");
    if (resource === "datasets" && !id && method === "GET") {
      const items = await store.list(user.id, "dataset");
      return json(items.map(({ records, passages, recordIds, ...dataset }) => ({ ...dataset, preview: records?.slice(0, 5) || passages?.slice(0, 3) })));
    }
    if ((resource === "datasets" || resource === "imports") && !id && method === "POST") {
      const input = await body(request);
      let filename: string, content: Buffer, source: string;
      if (resource === "imports") {
        if (typeof input.url !== "string") throw new LabError(400, "A public source URL is required.");
        const imported = await fetchSource(input.url);
        ({ filename, content, source } = imported);
      } else {
        if (typeof input.filename !== "string" || typeof input.content !== "string" || !/^[A-Za-z0-9+/]*={0,2}$/.test(input.content))
          throw new LabError(400, "Filename and base64 file bytes are required.");
        filename = input.filename.split(/[\\/]/).pop()!.slice(0, 120);
        content = Buffer.from(input.content, "base64"); source = filename;
        if (hash(content) !== input.sha256) throw new LabError(400, "Upload checksum mismatch; retry the file transfer.");
      }
      if (!content.length || content.length > LIMITS.uploadBytes) throw new LabError(413, "File must contain 1–2,000,000 bytes.");
      if ((await store.list(user.id, "dataset")).length >= LIMITS.datasetsPerOwner) throw new LabError(429, "This workspace reached its 30-dataset pilot envelope.");
      const retry = request.headers.get("Idempotency-Key") || "";
      const datasetId = "ds-" + hash(`${user.id}:${retry}`).slice(0, 32);
      const job = await store.enqueue(user.id, retry, { operation: "parse", datasetId, filename, source,
        content: content.toString("base64"), sha256: hash(content) });
      // Exact source snapshot is durable before any parsing/execution begins.
      await store.put(user.id, "source", datasetId, { filename, sha256: hash(content), content: content.toString("base64") });
      return json(publicJob(await startJob(user.id, job.id, store)), 202);
    }
    if (resource === "datasets" && id) {
      const dataset = await store.get(user.id, "dataset", id);
      if (method === "GET" && action === "source") {
        const source = await store.get(user.id, "source", id);
        return new Response(Buffer.from(source.content, "base64"), { headers: { ...headers, "Content-Type": "application/octet-stream", "Content-Disposition": `attachment; filename="${source.filename.replace(/[^\w. -]/g, "_")}"` } });
      }
      if (method === "GET") {
        const recordId = new URL(request.url).searchParams.get("record");
        if (recordId) {
          const index = dataset.recordIds?.indexOf(recordId);
          const passage = dataset.passages?.find((p: Document) => p.id === recordId);
          if ((index === undefined || index < 0) && !passage) throw new LabError(404, "Source reference does not resolve in this dataset.");
          return json({ source: dataset.source, sourceSha256: dataset.sha256, id: recordId, record: passage || dataset.records[index] });
        }
        return json(dataset);
      }
      if (method === "POST" && action === "confirm") {
        const mapping = await body(request, 20000);
        const job = await store.enqueue(user.id, request.headers.get("Idempotency-Key") || "", { operation: "confirm", datasetId: id, dataset, mapping });
        return json(publicJob(await startJob(user.id, job.id, store)), 202);
      }
      if (method === "POST" && action === "analyze") {
        if (!dataset.confirmed) throw new LabError(409, "Confirm the data interpretation before analysis.");
        const options = await body(request, 20000);
        if (!['unusual', 'forecast', 'patterns'].includes(options.mode)) throw new LabError(400, "Select a supported analysis.");
        if (options.mode === "forecast" && (dataset.kind === "documents" || dataset.confirmed.mode !== "temporal")) throw new LabError(400, "Forecasts require named numeric measurements and confirmed chronology.");
        const job = await store.enqueue(user.id, request.headers.get("Idempotency-Key") || "", { operation: "analyze", datasetId: id, dataset, mapping: dataset.confirmed, options });
        return json(publicJob(await startJob(user.id, job.id, store)), 202);
      }
    }
    if (resource === "jobs") {
      if (!id && method === "GET") return json(await store.jobs(user.id));
      if (id && method === "GET") return json(publicJob(await pollJob(user.id, id, store)));
      if (id && method === "POST" && action === "cancel") return json(publicJob(await cancelJob(user.id, id, store)));
      if (id && method === "POST" && action === "resume") return json(publicJob(await startJob(user.id, id, store)));
    }
    if (resource === "runs") {
      if (!id && method === "GET") return json((await store.list(user.id, "run")).map(run => ({ id: run.id, datasetId: run.datasetId, method: run.method, createdAt: run.createdAt, metrics: run.metrics, expired: run.expired })));
      const run = await store.get(user.id, "run", id);
      if (method === "GET" && action === "download") return new Response(JSON.stringify(run, null, 2), { headers: { ...headers, "Content-Type": "application/json", "Content-Disposition": `attachment; filename="${id}.json"` } });
      if (method === "GET") return json(run);
      if (method === "POST" && action === "feedback") {
        const input = await body(request, 5000);
        if (!run.findings.some((f: Document) => f.id === input.findingId) || !["expected", "useful", "false_alarm", "investigate"].includes(input.review)) throw new LabError(400, "Select a finding and a supported review outcome.");
        const review = { id: randomUUID(), runId: id, findingId: input.findingId, review: input.review,
          note: typeof input.note === "string" ? input.note.slice(0, 1000) : "", reviewer: user.id, reviewedAt: new Date().toISOString(),
          scope: "this finding; does not alter frozen labels, scores or model training" };
        await store.put(user.id, "feedback", review.id, review);
        return json(review, 201);
      }
      if (method === "POST" && action === "questions") {
        const input = await body(request, 5000);
        const dataset = await store.get(user.id, "dataset", run.datasetId);
        if (typeof input.question !== "string" || !input.question.trim() || input.question.length > 500) throw new LabError(400, "Enter a question of 1–500 characters.");
        if (run.encoder) {
          const job = await store.enqueue(user.id, request.headers.get("Idempotency-Key") || "", { operation: "retrieve", dataset, result: run, question: input.question });
          return json(publicJob(await startJob(user.id, job.id, store)), 202);
        }
        const finding = run.findings.find((f: Document) => f.id === input.findingId) || run.findings[0];
        const ownedReviews = await store.list(user.id, "feedback");
        const feedback = ownedReviews.filter(f => f.runId === id && f.findingId === finding?.id);
        const members: Document[] = finding?.members || [];
        const citations = members.slice(0, 20).map(member => {
          const index = dataset.recordIds.indexOf(member.recordId);
          if (index < 0) throw new LabError(502, "A finding reference failed source verification.");
          return { id: member.recordId, source: dataset.source, record: dataset.records[index], observed: member.actual ?? null,
            expected: member.expected ?? null, score: member.score, threshold: member.threshold, forecastId: member.forecastId || null };
        });
        return json({ method: "deterministic finding/evidence query; no language-model inference", question: input.question,
          answer: finding ? `${finding.what} ${finding.basis}. ${finding.whyItMatters} ${finding.nextAction}` : "This run has no flagged findings. That does not establish safety.",
          citations, reviews: feedback, grouping: finding?.grouping, limitations: finding?.uncertainty || run.limitations,
          similarHistoricalEvents: (await store.list(user.id, "run")).filter(other => other.id !== id).flatMap(other =>
            (other.findings || []).filter((f: Document) => f.entity === finding?.entity && f.detector === finding?.detector).slice(0, 3).map((f: Document) => ({ runId: other.id, findingId: f.id, basis: "same entity and detector; contextual similarity only",
              reviewedOutcomes: ownedReviews.filter(review => review.runId === other.id && review.findingId === f.id).map(review => ({ review: review.review, note: review.note, reviewedAt: review.reviewedAt, scope: review.scope })) }))).slice(0, 5) });
      }
    }
    if (resource === "feedback" && method === "GET") return json(await store.list(user.id, "feedback"));
    if (resource === "monitors") {
      if (!id && method === "GET") return json(await Promise.all((await store.list(user.id, "monitor")).map(m => monitorStatus(user.id, m.id, store))));
      if (!id && method === "POST") return json(await createMonitor(user.id, await body(request, 10000), store), 201);
      if (id && method === "GET") return json(await monitorStatus(user.id, id, store));
      if (id && method === "POST" && action === "process") return json(publicJob(await processMonitor(user.id, id, store)), 202);
      if (id && method === "POST" && action === "key") return json(await issueKey(user.id, id, store), 201);
      if (id && method === "POST" && action === "revoke") { await revokeKeys(user.id, id, store); return json({ revoked: true }); }
      if (id && method === "POST" && action === "reset") {
        const monitor = await store.get(user.id, "monitor", id);
        const reason = (await body(request, 2000)).reason;
        if (typeof reason !== "string" || reason.trim().length < 10) throw new LabError(400, "Record a reason for resetting learned state.");
        const resetId = randomUUID();
        await store.put(user.id, "checkpoint_archive", `${id}-${resetId}`, { checkpoint: monitor.checkpoint, monitorId: id, reason, resetAt: new Date().toISOString() });
        await store.put(user.id, "monitor", id, { ...monitor, checkpoint: { processedOffset: monitor.processedOffset }, resetId, resetReason: reason });
        return json({ resetId, reason });
      }
    }
    throw new LabError(404, "This Lab operation does not exist.");
  } catch (error) {
    const status = error instanceof LabError ? error.status : 500;
    if (status >= 500) console.error("GUIDED_API_FAILURE", { diagnosticId,
      errorType: error instanceof Error ? error.name : "unknown",
      code: error && typeof error === "object" && "code" in error ? String(error.code).slice(0, 80) : undefined });
    return json({ error: error instanceof LabError ? error.message : "The operation could not complete. Retry with the same job identity; diagnostic details are retained server-side.", diagnosticId }, status);
  }
}
