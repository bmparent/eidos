import { probePlaygroundGeneration } from '@/lib/works/playground-generation-probe';
export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=120;
export const POST=(request:Request)=>probePlaygroundGeneration(request);
