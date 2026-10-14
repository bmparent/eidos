# Observatory showcase: plain-language test analysis

The task was to make one piece of sensor evidence explain itself consistently across topology, a time series, a record list and its detail drawer. The same fixed time window, host and record-type filters now govern these views. Switching views preserves selection. The download carries source records and selection for independent inspection.

All 122 Node tests passed. Canonical Brain parity passed 954 frames and 41,022 comparisons. The Python suite passed 27 cases; an existing symlink test failed before exercising its assertion because Windows denied symlink creation. That original failure remains in the saved log.

Rendered checks use clearly labeled synthetic records. They exercise duplicate delivery, selection, view changes, time filtering, keyboard host filtering, download, malformed and valid imports, model fallback and denied/reconnected feeds. The final browser receipt records exact outcomes. Screenshots cover desktop and 390-pixel width with reduced motion and GPU disabled. This verifies a useful SVG/HTML fallback, not hardware GPU performance.

Local artifacts are `artifacts/showcase-observatory-20261013`. The independent download is `fixture-export.json`. Drive status is in `drive_manifest.json`; a mounted copy alone does not prove remote synchronization. The existing owner-private Observatory and its sensor/model configuration were preserved. The new private candidate has its own identity, with deployment and unavailable bindings documented separately.

## Proof Logic + Meaning

The local evidence-inspection gate passed; hosted/provider acceptance stays separate. Before this change, the topology and filtered event list could show different evidence populations. The new logic deduplicates stable IDs, retains the newest 2,000 receipts, and applies one intersection: selected time AND selected host AND selected kind. Time-bin counts count retained records; they do not estimate bandwidth or attacks. Missing intervals stay unknown.

This is explanation before automation and truth that can be revisited. It improves operator trust by making counts, selected receipts and coverage agree. It strengthens the north-star system's human-readable incident receipts and reproducibility. The test/browser/export/build receipts support that narrow result; compression advantage, detection precision/recall, actual model inference and broader domain generalization remain unproven. No percentage or scientific progress score was invented. Core model and sensor behavior did not change.

Next: review the exact isolated hosted candidate, then attach a scoped real source only when its required configuration and acceptance are available. The Windows symlink test should be rerun in an environment that permits symlinks with the exact Python command in the acceptance report.

## Hosted candidate acceptance update

The separate owner-private candidate deployed successfully and passed seven hosted rendered fixture checks at https://sentinel-showcase-candidate-20261013.bmparent.chatgpt.site/evidence-fixture. One owner, no external visitors. The original Site was untouched. No sensor DB/model/IP relay was copied; missing source and model availability are explicit, so no actual sensor/model execution claim follows. See deployment.json and hosted-browser-acceptance.json. Failed auth-header/status assumptions and the Windows packaging limitation are preserved in the acceptance record.
