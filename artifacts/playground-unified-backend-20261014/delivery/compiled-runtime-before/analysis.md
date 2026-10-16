# Cross-build archive identity defect — retained negative evidence

The source-level tests ran inside one TypeScript loader. They proved internal snapshot/ZIP consistency but did not prove that Vite and Next produce the same serialized runtime. The browser integration exposed that missing boundary.

A plain Node harness loaded the actual optimized Next 16.3.8 route and its compiled delivery module (11160), called its freeze/read functions using real in-memory SQLite, and blocked every outbound fetch. It reused only SNAPSHOT.json from the controlled frontend fixture archive. The actual production bundle contains no `__name` helper and its generated script passes JavaScript parsing. The `tsx` helper failure is a separate fixture issue.

The exact same snapshot nevertheless yields different ZIPs across compilers. Only `script.js` and its entry in `PACKAGE_MANIFEST.json` differ: 23,059 fixture script bytes versus 20,240 production Next script bytes. Both input/output archives and hashes are retained here. The browser agent independently observed the Vite/local versus server mismatch. The product correctly rejected that mismatch; its exact-byte gate was not bypassed.

Cause: `pageRuntime.toString()` serializes compiler-specific source text. The fix must provide a canonical runtime string shared by every build, with a regression that compares actual cross-build archive bytes. Passing source-loader tests alone cannot close this gate.

The current second manual preview remains feature-disabled, has no database or paid archives, and cannot perform public delivery. No deployment or source change was made by this probe. Root assigned the deterministic renderer fix before the next isolated preview.
