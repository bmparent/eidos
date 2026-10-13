import {createClient} from '@libsql/client';
import {createHash} from 'node:crypto';

// This seed is pinned to the validation database recorded on 2026-09-23.
// It never overwrites an owner-edited item and refuses every other database.
const expectedUrlHash='dbf0ee4f912f8b322b63005d95e42948cac719b35b880e48ef18c491a236f237';
const expectedTokenHash='60dae48deccd61a154da6b5802b43eab70286d810cedea7e2064d9b10fb25b1a';
const hash=value=>createHash('sha256').update(value||'').digest('hex');
if(process.argv[2]!=='--validation' || hash(process.env.EIDOS_DATABASE_URL)!==expectedUrlHash || hash(process.env.EIDOS_DATABASE_AUTH_TOKEN)!==expectedTokenHash)
  throw Error('The exact recorded validation database and --validation are required.');

const evidence='https://github.com/bmparent/brent-parent-intelligence-studio/blob/codex/works-audit-implementation-20260921/docs/implementation/ACCEPTANCE_2026-09-23.md';
const items=[
  ['release-google','Google preview callback and real consent','Callback is absent from the provider client; real consent and account boundary checks remain open.','high'],
  ['release-inbox','Verify actual inbox receipt','HTTP and provider acceptance do not prove delivery to the intended inbox.','high'],
  ['release-stripe-test','Complete new Stripe TEST purchase and recovery','New checkout, signed webhook, exact archive, refund or dispute are not accepted yet.','high'],
  ['release-cloud-editor','Accept authenticated cloud editor and AI flow','Save/reopen, provider proposal, apply/Undo, and interruption still need a paired hosted session.','high'],
  ['release-device','Native accessibility and 200% zoom','Physical phone, screen reader, and 200% zoom remain open.','normal'],
  ['release-snapshot','Snapshot security, cost and storage','Pinned outbound capture, budget, storage lifecycle, and recovery remain blocked; keep off.','critical'],
  ['reported-account','Investigate reported account identity mismatch','Reported username and linked identity mismatch. Do not merge accounts by similarity; reproduce with controlled test identities.','high'],
  ['reported-save','Investigate reported cloud save/reopen issue','Reported save or reopen problem is not reproduced in the paired authenticated preview.','high'],
];
const client=createClient({url:process.env.EIDOS_DATABASE_URL,authToken:process.env.EIDOS_DATABASE_AUTH_TOKEN});
try{
  const now=new Date().toISOString();
  for(const [id,title,detail,severity] of items)
    await client.execute({sql:'INSERT OR IGNORE INTO eidos_ops_work(id,title,detail,status,severity,evidence_url,due_at,source,created_at,updated_at,version) VALUES(?,?,?,?,?,?,?,?,?,?,1)',args:[id,title,detail,'blocked',severity,evidence,null,'acceptance-2026-09-23',now,now]});
  const count=await client.execute("SELECT COUNT(*) AS n FROM eidos_ops_work WHERE source='acceptance-2026-09-23'");
  console.log(`Validation work items present: ${count.rows[0].n}`);
}finally{client.close()}
