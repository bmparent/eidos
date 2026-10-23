import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { dispatchWorks } from '../lib/works/dispatch';

test('Public profile reads cross the authenticated relay on real libSQL without exposing another owner or unpublished activity', async () => {
  const client = createClient({ url: ':memory:' });
  try {
    for (const migration of ['0001_eidos_platform.sql', '0002_members.sql'])
      await client.executeMultiple(readFileSync('lib/works/vendor/migrations/' + migration, 'utf8'));
    const now = '2026-10-23T12:00:00.000Z';
    for (const [id, username] of [['owner-alice', 'qa_alice'], ['owner-bob', 'qa_bob']])
      await client.execute({ sql: "INSERT INTO eidos_email_members(id,email,username,kind,created_at,newsletter_after) VALUES(?,?,?,'person',?,?)", args: [id, username + '@example.test', username, now, now] });
    const publicId = crypto.randomUUID(), hiddenId = crypto.randomUUID(), otherId = crypto.randomUUID();
    for (const [id, owner, status] of [[publicId, 'owner-alice', 'published'], [hiddenId, 'owner-alice', 'pending'], [otherId, 'owner-bob', 'published']])
      await client.execute({ sql: "INSERT INTO eidos_threads(id,title,body,category,author,author_type,owner_id,status,created_at) VALUES(?,'Shared work','A useful work update','build','@qa_alice','guest',?,?,?)", args: [id, owner, status, now] });
    for (const [id, threadId, status] of [['visible', publicId, 'published'], ['rejected', publicId, 'rejected'], ['hidden-parent', hiddenId, 'published']])
      await client.execute({ sql: "INSERT INTO eidos_replies(id,thread_id,body,author,author_type,owner_id,status,created_at) VALUES(?,?,'A useful reply','@qa_alice','guest','owner-alice',?,?)", args: [id, threadId, status, now] });
    const source = { EIDOS_PLATFORM_TOKEN: 'synthetic-relay-token-at-least-32-characters', PUBLIC_SITE_URL: 'https://eidos-works.com' };
    const env = { EIDOS_DB: adaptDatabase(client), EIDOS_ACCOUNTS_ENABLED: 'false' };
    const call = (path: string, token = source.EIDOS_PLATFORM_TOKEN) => dispatchWorks(new Request('https://backend.vercel.app/api/works/v1' + path, {
      headers: { 'x-eidos-platform-token': token, 'x-eidos-site-origin': source.PUBLIC_SITE_URL },
    }), source, env);
    const response = await call('/api/members/directory?username=qa_alice');
    assert.equal(response.status, 200, await response.clone().text());
    const profile = await response.json();
    assert.deepEqual(profile.posts.map((p: {id:string}) => p.id), [publicId]);
    assert.deepEqual(profile.replies.map((r: {id:string}) => r.id), ['visible']);
    assert.deepEqual(Object.keys(profile.member).sort(), ['created_at', 'kind', 'username']);
    assert.doesNotMatch(JSON.stringify(profile), /owner-alice|owner-bob|example\.test|newsletter|hidden-parent|rejected/);
    assert.equal((await call('/api/members/directory?username=qa_alice', 'wrong')).status, 401);
    const thread = await (await call('/api/community/threads?id=' + publicId)).json();
    assert.equal(thread.thread.member_username, 'qa_alice');
    assert.equal(thread.replies[0].member_username, 'qa_alice');
    const html = await (await call('/community/thread/' + publicId)).text();
    assert.match(html, /href="\/members\/qa_alice"/);
    await client.execute("UPDATE eidos_email_members SET disabled=1 WHERE username='qa_alice'");
    assert.equal((await call('/api/members/directory?username=qa_alice')).status, 404);
  } finally { client.close(); }
});
