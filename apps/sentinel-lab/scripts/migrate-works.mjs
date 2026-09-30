import { createClient } from '@libsql/client';
import { readFile, readdir } from 'node:fs/promises';

const url = process.env.EIDOS_DATABASE_URL;
if (!url || !process.env.EIDOS_DATABASE_AUTH_TOKEN || !['libsql:', 'https:'].includes(new URL(url).protocol))
  throw Error('Configure the dedicated remote Eidos Works database before migrating.');
const client = createClient({ url, authToken: process.env.EIDOS_DATABASE_AUTH_TOKEN });
try {
  const directory = new URL('../lib/works/vendor/migrations/', import.meta.url);
  for (const name of (await readdir(directory)).filter(n=>/^\d+.*\.sql$/.test(n)).sort()) {
    const schema = await readFile(new URL(name,directory),'utf8');
    // SQLite migrations include triggers and quoted semicolons. The SQL parser,
    // rather than string splitting, must find the statement boundaries.
    await client.executeMultiple(schema);
  }
  console.log('Applied the additive Eidos Works schema.');
} finally { client.close(); }
