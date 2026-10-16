import { dispatchWorks } from '@/lib/works/dispatch';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
// The bounded Playground provider request may take 90 seconds; leave time for
// durable accounting and the 105-second Pages relay before ending this route.
export const maxDuration = 120;
export const GET = (request: Request) => dispatchWorks(request);
export const POST = (request: Request) => dispatchWorks(request);
