# Playground authenticated TEST continuation

The separate protected Preview diagnostic checks the dedicated branch key using only `GET https://api.openai.com/v1/models`. It returns whether the two requested model IDs are present, the provider request ID and frozen source identity. It does not run generation or counting, reveal provider errors or credentials, modify database rows, change the manual preview alias, or activate accounts/commerce. Production returns 404 even if its flag is set. Preview additionally requires `EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS=test-auth`, the existing private relay token and exact Playground site origin.

The approved key stays in Vercel as `EIDOS_PLAYGROUND_OPENAI_API_KEY`, scoped to Preview branch `codex/playground-unified-20261014`. The deploy helper inherits it server-side and repeats all other isolated disabled overrides. No local OpenAI key file is needed.

```text
node --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs --test apps/sentinel-lab/tests/playground-provider-check.test.ts
npm --prefix apps/sentinel-lab test
npm --prefix apps/sentinel-lab run lint
npm --prefix apps/sentinel-lab run build
node apps/sentinel-lab/scripts/deploy-playground-auth-check.mjs --cli=<authenticated-vercel-cli-entry> --deploy
```

Receipts go to `artifacts/playground-authenticated-20261015/`. A successful metadata request proves credential authentication and listed model access; inference, actual usage, token-count billing, owner isolation, TEST funding and release acceptance require their own observations.

An existing ignored configuration was found in the retained paid-template backend worktree. A read-only connection to its explicitly selected `EIDOS_TEMPLATES_TEST` validation database succeeded. Its contents remain private. No credentials were read from the Vercel Secret API and no other branch bindings were changed.
