export class RequestBodyError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export async function readExperimentJson(request: Request) {
  const limit = 65_536;
  if (Number(request.headers.get("content-length")) > limit) throw new RequestBodyError("Experiment request exceeds 64 KiB.", 413);
  const reader = request.body?.getReader();
  if (!reader) throw new RequestBodyError("An experiment JSON body is required.", 400);
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) {
        await reader.cancel();
        throw new RequestBodyError("Experiment request exceeds 64 KiB.", 413);
      }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  try { return JSON.parse(Buffer.concat(chunks).toString("utf8")); }
  catch { throw new RequestBodyError("Send a valid experiment JSON body.", 400); }
}
