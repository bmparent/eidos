import test from 'node:test';
import assert from 'node:assert/strict';
import { observeStripeTest } from '../lib/works/stripeObservation';

test('Stripe observation is TEST-only and redacts its key', async () => {
  const unavailable = await observeStripeTest('sk_live_example',[],[],async () => { throw Error('must not fetch'); });
  assert.equal(unavailable.status,'not_configured');
  const calls:string[]=[];
  const fetcher = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url=String(input);calls.push(url);
    assert.equal(init?.headers && (init.headers as Record<string,string>).authorization,'Bearer sk_test_example');
    const value = url.includes('/payment_intents/') ? {id:'pi_123',status:'succeeded'} : {data:[]};
    return new Response(JSON.stringify(value),{status:200});
  };
  const observed=await observeStripeTest('sk_test_example',[{id:'kit-1',status:'paid',payment_intent:'pi_123'}],[],fetcher as typeof fetch,new Date('2026-09-24T12:00:00Z'));
  assert.equal(observed.status,'healthy');
  assert.equal(observed.data?.mismatchCount,0);
  assert.equal(observed.data?.rows[0].providerStatus,'succeeded');
  assert.equal(observed.data?.webhookDelivery,'unavailable');
  assert.equal(JSON.stringify(observed).includes('sk_test_example'),false);
  assert.equal(calls.length,3);
});

test('Stripe API failure is unavailable rather than an empty successful read', async () => {
  const observed=await observeStripeTest('sk_test_example',[],[],async () => new Response('{}',{status:403}));
  assert.equal(observed.status,'unavailable');
  assert.equal(observed.errorCode,'stripe_http_403');
  assert.equal(observed.data,null);
});
