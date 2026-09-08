import { db, HttpError, reserve, type PlatformEnv } from './core';

export type Mail = {
  to: string;
  subject: string;
  text: string;
  html?: string;
  headers?: Record<string, string>;
};
/** One recipient per message; API credentials and provider bodies never reach clients. */
export async function sendMemberMail(
  env: PlatformEnv,
  mail: Mail,
  idempotencyKey: string,
) {
  if (!env.RESEND_API_KEY || !env.EIDOS_MAIL_FROM)
    throw new HttpError(
      503,
      'Email delivery is temporarily unavailable. Please try again later.',
    );
  const limit = Math.min(
    1000,
    Math.max(1, Number(env.EIDOS_MAIL_DAILY_LIMIT) || 90),
  );
  if (!(await reserve(db(env), 'member-mail-global', 1, limit)))
    throw new HttpError(
      429,
      'Today’s email allowance has been reached. Please try again tomorrow.',
    );
  const response = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      authorization: `Bearer ${env.RESEND_API_KEY}`,
      'content-type': 'application/json',
      'idempotency-key': idempotencyKey,
    },
    body: JSON.stringify({ from: env.EIDOS_MAIL_FROM, ...mail }),
    redirect: 'error',
    signal: AbortSignal.timeout(8000),
  });
  const result = (await response.json().catch(() => null)) as {
    id?: string;
  } | null;
  if (!response.ok || !result?.id)
    throw new HttpError(
      503,
      'Email delivery is temporarily unavailable. Please try again later.',
    );
  return result.id;
}
