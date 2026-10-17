import test from 'node:test';
import assert from 'node:assert/strict';
import {assertCloudStudioBoundary, LOCAL_STUDIO_CLOUD_NOTICE} from '../lib/works/vendor/functions/_shared/platform/studioBoundary';
import {saveProject} from '../lib/works/vendor/functions/_shared/platform/playground';
import {onRequestPost} from '../lib/works/vendor/functions/api/playground/projects';
import {createProject} from '../lib/works/vendor/src/playground/model';
import type {PlatformEnv} from '../lib/works/vendor/functions/_shared/platform/core';

test('studio documents and disguised studio fields cannot be downgraded through the schema-1 cloud contract', async () => {
  for(const document of [{...createProject(),schemaVersion:5}, {...createProject(),studios:{}}, {...createProject(),studios:null}, {...createProject(),schemaVersion:6}]){
    const snapshot=JSON.stringify(document);let databaseTouches=0;
    const env=Object.defineProperty({},'EIDOS_DB',{get(){databaseTouches++;throw Error('Cloud boundary must run before database access');}}) as PlatformEnv;
    await assert.rejects(saveProject(env,'owner',{document}),error=>error instanceof Error&&'status'in error&&error.status===409&&error.message===LOCAL_STUDIO_CLOUD_NOTICE);
    assert.equal(databaseTouches,0);assert.equal(JSON.stringify(document),snapshot);
  }
});
test('HTTP studio rejection precedes auth/schema database access and explains the local backup path', async () => {
  let databaseTouches=0;
  const env=Object.defineProperty({PUBLIC_SITE_URL:'https://eidos-works.com'},'EIDOS_DB',{get(){databaseTouches++;throw Error('Unexpected database access');}}) as PlatformEnv;
  const response=await onRequestPost({env,request:new Request('https://eidos-works.com/api/playground/projects',{method:'POST',headers:{origin:'https://eidos-works.com','content-type':'application/json'},body:JSON.stringify({document:{schemaVersion:5,studios:{}}})})});
  assert.equal(response.status,409);assert.match(await response.text(),/project JSON backup/);assert.equal(databaseTouches,0);
});
test('legacy schema-1 acceptance and malformed-input validation remain delegated to their existing contracts', () => {
  const legacy=createProject(),snapshot=JSON.stringify(legacy);
  assert.doesNotThrow(()=>assertCloudStudioBoundary(legacy));assert.equal(JSON.stringify(legacy),snapshot);
  for(const value of [null,undefined,[],{},'invalid'])assert.doesNotThrow(()=>assertCloudStudioBoundary(value));
});
