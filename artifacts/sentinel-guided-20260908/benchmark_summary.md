# Frozen synthetic final evaluation

| Method | MAE | Incident precision | Incident recall | False alerts/day | Coverage |
|---|---:|---:|---:|---:|---:|
| eidos_adapt_all | 0.9831 | 0.6000 | 0.8000 | 18.4615 | 0.8767 |
| eidos_multiscale | 0.9723 | 0.6000 | 0.8000 | 17.5824 | 0.8602 |
| eidos_none | 0.9866 | 0.6000 | 0.8000 | 16.7033 | 0.8578 |
| eidos_regulation | 1.0072 | 0.5556 | 0.8000 | 18.4615 | 0.8443 |
| isolation_forest_prefix | NA | 0.2875 | 0.5333 | 38.6813 | NA |
| persistence | 1.0326 | 0.6000 | 0.8000 | 14.5055 | 0.8541 |
| robust_prefix | NA | 0.8333 | 1.0000 | 14.0659 | NA |
| seasonal_12 | 1.8446 | 0.5509 | 1.0000 | 24.6154 | 0.8132 |

These are means of whole-seed period means across three independent generated periods; the seven scenarios within a period are correlated. Min/max period ranges, interval widths, detection delay and raw/merged event counts remain in final-summary.json, final-metrics.csv and the verified raw ZIP. They are not iid confidence intervals. NA means the method does not issue forecasts or interval estimates. Every Eidos variant failed the prespecified operational qualification; all remain experimental. The default MAE improves over persistence here but its false-alert burden is higher. The robust prefix baseline has better incident precision/recall in this fixture set; it also fails the one-false-alert-per-day limit. No baseline is promoted as an operational guarantee.

Memory was a canonical Hippocampus shadow observer, with suppression deliberately disabled; unchanged detection does not establish utility. TraceSeal changed shadow scores but did not establish a safety/accuracy benefit. Neither is enabled as a proven improvement. Peak local resident memory: 364,937,216 bytes. Hosted peak memory and attributed infrastructure dollars are unknown; 8 GB is an allocation limit, not a measurement. LLM calls and LLM cost are zero.
