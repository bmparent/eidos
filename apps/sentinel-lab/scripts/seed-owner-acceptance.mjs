import { createClient } from '@libsql/client';
import { createHash, randomBytes } from 'node:crypto';

// Synthetic identities only, and only in the pinned validation database.
const urlHash = 'dbf0ee4f912f8b322b63005d95e42948cac719b35b880e48ef18c491a236f237';
const tokenHash = '60dae48deccd61a154da6b5802b43eab70286d810cedea7e2064d9b10fb25b1a';
const hash = value => createHash('sha256').update(value || '').digest('hex');
if (process.argv[2] !== '--validation' || hash(process.env.EIDOS_DATABASE_URL) !== urlHash || hash(process.env.EIDOS_DATABASE_AUTH_TOKEN) !== tokenHash)
  throw Error('Exact recorded validation database and --validation required.');

const identities = [
  { id: 'owner-console-acceptance-a-20260924', email: 'owner-console-a@example.invalid', username: 'ownerconsoleqa_a' },
  { id: 'owner-console-acceptance-b-20260924', email: 'owner-console-b@example.invalid', username: 'ownerconsoleqa_b' },
];
const client = createClient({ url: process.env.EIDOS_DATABASE_URL, authToken: process.env.EIDOS_DATABASE_AUTH_TOKEN });
try {
  const now = new Date().toISOString();
  for (const person of identities) {
    await client.execute({ sql: 'INSERT OR IGNORE INTO eidos_email_members(id,email,username,kind,created_at,newsletter_after,disabled) VALUES(?,?,?,\'person\',?,?,0)',
      args: [person.id,person.email,person.username,now,now] });
    const record = await client.execute({ sql: 'SELECT email,kind FROM eidos_email_members WHERE id=?',args:[person.id] });
    if (record.rows[0]?.email !== person.email || record.rows[0]?.kind !== 'person') throw Error('Controlled identity collision.');
    const sessions = await client.execute({ sql:'SELECT COUNT(*) AS n FROM eidos_member_sessions WHERE member_id=?',args:[person.id] });
    if (Number(sessions.rows[0].n) === 0) {
      const credential = randomBytes(32);
      const credentialHash = createHash('sha256').update(credential).digest('hex');
      credential.fill(0);
      await client.execute({ sql:'INSERT INTO eidos_member_sessions(token_hash,member_id,expires) VALUES(?,?,?)',args:[credentialHash,person.id,Date.now()+4*60*60*1000] });
    }
  }
  console.log(JSON.stringify({ validationOnly:true, syntheticMemberIds:identities.map(x=>x.id), providerOrdersCreated:0, realMemberRowsChanged:0 }));
} finally { client.close(); }
