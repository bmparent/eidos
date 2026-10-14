import { createTargetObservationHandler } from "../../../../../sentinel-observatory/live/node-probes.mjs";

export const runtime = "nodejs";
export const maxDuration = 20;

const observe = createTargetObservationHandler();
export const POST = observe;
