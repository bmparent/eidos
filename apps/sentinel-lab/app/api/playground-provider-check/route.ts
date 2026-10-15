import { checkPlaygroundProvider } from '@/lib/works/playground-provider-check';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 30;
export const GET = (request: Request) => checkPlaygroundProvider(request);
