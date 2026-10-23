import { createClient } from '@libsql/client';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const options = Object.fromEntries(process.argv.slice(2).map(value => value.replace(/^--/, '').split('=')));
const url = process.env.EIDOS_DATABASE_URL, token = process.env.EIDOS_DATABASE_AUTH_TOKEN;
if (!url || !token || !options['expected-host'] || !options.out) throw Error('Supply configured validation database, --expected-host=... and --out=...');
const destination = new URL(url);
if (!['libsql:', 'https:'].includes(destination.protocol) || destination.username || destination.password || destination.hostname !== options['expected-host'] || !destination.hostname.includes('validation'))
  throw Error('Refusing a database outside the explicit validation destination.');
const client = createClient({url,authToken:token});
try {
  const before = (await client.execute('SELECT COUNT(*) count FROM eidos_orders')).rows[0].count;
  const schema = await readFile(new URL('../lib/works/vendor/migrations/0004_paid_templates.sql', import.meta.url), 'utf8');
  await client.executeMultiple(schema);
  const after = (await client.execute('SELECT COUNT(*) count FROM eidos_orders')).rows[0].count;
  if (before !== after) throw Error('Existing order count changed during additive schema setup. Inspect before proceeding.');
  const receipt = { timestampUtc:new Date().toISOString(), databaseHost:destination.hostname, migration:'0004_paid_templates.sql', mode:'TEST candidate validation only', existingOrderCountBefore:before, existingOrderCountAfter:after, preservedExistingOrders:true };
  const output = path.resolve(options.out);await mkdir(path.dirname(output),{recursive:true});await writeFile(output,JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify(receipt));
} finally {client.close();}
