import { createIpObservationHandler } from "@/lib/observatory/ip-observer";

export const runtime = "nodejs";
export const maxDuration = 15;

// Only the server-side authenticated Observatory relay gets the bearer secret.
// The handler fails closed until OBSERVATORY_COLLECTOR_SECRET is configured.
const observe = createIpObservationHandler();
export const POST = observe;
