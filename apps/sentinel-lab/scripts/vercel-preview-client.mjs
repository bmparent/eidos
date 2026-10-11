import { spawnSync } from 'node:child_process';

/** Use the authenticated CLI; never read or print its personal authentication file. */
export function vercelApi(cli, endpoint, method = 'GET', payload) {
  const args = [cli, 'api', endpoint, '--method', method, '--raw', '--non-interactive'];
  if (payload !== undefined) args.push('--input', '-');
  const result = spawnSync(process.execPath, args, { encoding: 'utf8', input: payload === undefined ? undefined : JSON.stringify(payload), maxBuffer: 24 * 1024 * 1024, timeout: 180000 });
  let parsed;
  try { parsed = JSON.parse(result.stdout || '{}'); } catch { throw Error('Vercel CLI did not return a JSON API response. Inspect authenticated CLI access.'); }
  if (result.status !== 0 || parsed.error) throw Error('Vercel API rejected the request: ' + (parsed.error?.code || 'cli_error'));
  return parsed;
}
