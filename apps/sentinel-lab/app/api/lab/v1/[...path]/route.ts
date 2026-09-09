import { handleGuided } from "@/lib/guided/api";

export const runtime = "nodejs";
export const maxDuration = 300;
const handle = async (request: Request, context: { params: Promise<{ path: string[] }> }) =>
  handleGuided(request, (await context.params).path);
export { handle as GET, handle as POST, handle as DELETE };
