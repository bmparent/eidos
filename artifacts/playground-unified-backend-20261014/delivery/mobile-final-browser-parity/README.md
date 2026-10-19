# Final mobile source: actual Next versus raw browser package

From the backend repository root at deployed source `bd3c6ab2a636a5a1783babd304f2ac4c75f7a1b6`:

```text
npm --prefix apps/sentinel-lab run build
node artifacts/playground-unified-backend-20261014/delivery/mobile-final-browser-parity/probe.cjs artifacts/playground-unified-backend-20261014/delivery/mobile-final-browser-parity/browser-approved.zip --require-match
```

The plain Node probe loads the actual optimized Next route module, blocks every outbound fetch, freezes the input SNAPSHOT.json in real in-memory SQLite and reads its immutable ZIP back. It requires the complete result to equal the raw package saved by the built Vite browser, including the uploaded image. It also parses script.js and rejects the injected __name helper seen in an earlier fixture.

Both packages contain 128,380 bytes and SHA256 `9def4cc49c1f79a061af9470c46fb96d6aa3550ba490304770ea1550364f29ed`. Probe outputs go to ignored `apps/sentinel-lab/.vercel/compiled-delivery-check/`; this retained receipt is not overwritten. No hosted database, provider, payment or mail is involved. This is an exact source-specific fixture, not acceptance of hosted customer persistence.
