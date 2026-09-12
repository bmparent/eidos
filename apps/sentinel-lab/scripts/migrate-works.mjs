import { createClient } from '@libsql/client';
import { readFile, readdir } from 'node:fs/promises';

const prefix = process.env.EIDOS_DATABASE_BINDING_PREFIX;
if (prefix && !/^EIDOS_[A-Z0-9_]{1,48}$/.test(prefix)) throw Error('Invalid Works database binding prefix');
const url = prefix ? process.env[prefix + '_TURSO_DATABASE_URL'] : process.env.EIDOS_DATABASE_URL;
const authToken = prefix ? process.env[prefix + '_TURSO_AUTH_TOKEN'] : process.env.EIDOS_DATABASE_AUTH_TOKEN;
if (!url || !authToken || !['libsql:', 'https:'].includes(new URL(url).protocol))
  throw Error('Configure the dedicated remote Eidos Works database before migrating.');
const client = createClient({ url, authToken });
try {
  const directory = new URL('../lib/works/vendor/migrations/', import.meta.url);
  for (const name of (await readdir(directory)).filter(n=>/^\d+.*\.sql$/.test(n)).sort()) {
    const schema = await readFile(new URL(name,directory),'utf8');
    await client.batch(name === '0004_playground_asset_quota.sql' ? [schema] : schema.split(';').map(s => s.trim()).filter(Boolean), 'write');
    console.log('Applied additive migration: ' + name);
  }
  console.log('Applied the additive Eidos Works schema.');
} finally { client.close(); }
