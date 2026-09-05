import { createClient } from '@libsql/client';
import { readFile } from 'node:fs/promises';

const url = process.env.EIDOS_DATABASE_URL;
if (!url || !process.env.EIDOS_DATABASE_AUTH_TOKEN || !['libsql:', 'https:'].includes(new URL(url).protocol))
  throw Error('Configure the dedicated remote Eidos Works database before migrating.');
const client = createClient({ url, authToken: process.env.EIDOS_DATABASE_AUTH_TOKEN });
try {
  const schema = await readFile(new URL('../lib/works/vendor/migrations/0001_eidos_platform.sql', import.meta.url), 'utf8');
  await client.batch(schema.split(';').map(s => s.trim()).filter(Boolean), 'write');
  console.log('Applied the additive Eidos Works schema.');
} finally { client.close(); }
