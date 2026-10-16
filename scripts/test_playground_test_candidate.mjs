import test from 'node:test';
import assert from 'node:assert/strict';
import { testEnvironment, testSite } from './deploy_playground_test_candidate.mjs';
const source={EIDOS_DATABASE_BINDING_PREFIX:'EIDOS_TEMPLATES_TEST',EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL:'libsql://owned-validation.example.test',STRIPE_SECRET_KEY:'sk_test_fixture',RESEND_API_KEY:'must-inherit',EIDOS_PLAYGROUND_OPENAI_API_KEY:'must-inherit',OPENAI_API_KEY:'never-use',EIDOS_TEMPLATE_LIVE_ENABLED:'true'};
test('TEST candidate scopes accounts and checkout while inheriting only authorized mail/provider secrets',()=>{
  const env=testEnvironment(source,'r'.repeat(32));
  assert.equal(env.PUBLIC_SITE_URL,testSite);assert.equal(env.EIDOS_PREVIEW_ORIGINS,testSite);
  assert.equal(env.EIDOS_TEMPLATE_LIVE_ENABLED,'false');assert.equal(env.EIDOS_LOCAL_TEST,'false');
  assert.equal(env.EIDOS_AI_ENABLED,'false');assert.equal(env.EIDOS_PLAYGROUND_AI_ENABLED,'false');assert.equal(env.EIDOS_PLAYGROUND_INPUT_COUNT_PRICING,'');
  assert.equal(env.EIDOS_ACCOUNTS_ENABLED,'true');assert.equal(env.EIDOS_TEMPLATE_TEST_ENABLED,'true');
  assert.equal(env.EIDOS_PLAYGROUND_DELIVERY_MODE,'test');assert.equal(env.OPENAI_API_KEY,'');
  assert.equal('RESEND_API_KEY' in env,false);assert.equal('EIDOS_PLAYGROUND_OPENAI_API_KEY' in env,false);
  assert.equal(source.EIDOS_TEMPLATE_LIVE_ENABLED,'true');
});
test('TEST candidate rejects another database, LIVE provider key and missing relay before writes',()=>{
  for(const change of [{EIDOS_DATABASE_BINDING_PREFIX:'PRODUCTION'},{EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL:'libsql://production.example.test'},{STRIPE_SECRET_KEY:'sk_live_fixture'}])assert.throws(()=>testEnvironment({...source,...change},'r'.repeat(32)));
  assert.throws(()=>testEnvironment(source,''));
});
