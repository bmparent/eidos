import { dispatchWorks } from '@/lib/works/dispatch';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 30;
export const GET = (request: Request) => dispatchWorks(request);
export const POST = (request: Request) => dispatchWorks(request);
