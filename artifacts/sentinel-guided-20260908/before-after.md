# Observed before/after comparison

| Aspect | Before | Implemented candidate | Evidence |
|---|---|---|---|
| Main flow | Research console/Kaggle and recorded engine view | Add data, confirm schema, analyze, investigate, saved work and monitors | preview-final/before-production.png and 01–07 screenshots |
| Forecast interpretation | Predictor chosen using current residual | Prediction committed before target; historical-loss selection, prior baseline and named units | semantic-final/receipt.json, runner-final.xml, evaluation raw issuance |
| Long text | Equal first 64 characters produce equal legacy vectors | Full-token semantic vectors differ by L2 1.2528; paraphrase cosine .7783 vs unrelated .0154 | semantic-final/receipt.json |
| Final synthetic comparison | Persistence MAE 1.0326, false alerts/day 14.5055 | Default Eidos MAE .9866, false alerts/day 16.7033; precision .60 | evaluation/final-summary.json |
| Hosted service fixture | Persistence MAE 2.2658 | Eidos MAE 2.8254; Eidos loses here | preview-final/browser-result.json |
| Stream recovery | No general scoped monitor path | Exact checkpoint state matches uninterrupted real Torch replay; ingress/revocation verified in preview | stream/receipt.json; stream-release-local/receipt.json; preview-budget/receipt.json |

All screenshots are actual browser captures. These comparisons are engineering and synthetic-period evidence, not broad operational qualification. Full final-SHA hosted compute remains blocked by Vercel HTTP 402. Legacy engine code and sealed proof data were preserved.
