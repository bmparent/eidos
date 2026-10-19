import { checkPlaygroundProvider } from '@/lib/works/playground-provider-check';
import { probePlaygroundCount } from '@/lib/works/playground-count-probe';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 30;
export const GET = (request: Request) => checkPlaygroundProvider(request);
export const POST = (request: Request) => probePlaygroundCount(request);
