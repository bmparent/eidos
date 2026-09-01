# Eidos Brain / Eidos Sentinel — Meaningful Surprise v1

**Status:** design-freeze contract
**Protocol family:** `EIDOS-MS-v1`
**Freeze date:** 2026-09-01
**Scope:** Eidos Brain / Eidos Sentinel only
**Repository baseline inspected:** `00489e358865994ec4b40a5c5bfdfa034560773a`
**Main baseline:** `f676fe2342b98886bc04cd8f4b0e943fce77ec9a`

## 1. Purpose

Eidos should not call an observation meaningful merely because it is difficult to predict, rare, geometrically unusual, familiar, or expensive to compress. Those are evidence channels. Meaning is the expected consequence of observing and preserving the event for a bounded task.

The v1 center is therefore:

> A deviation is meaningful when observing it can change the best bounded action or materially reduce expected task loss. Eidos should select a representation, storage fidelity, and alert action that preserve that decision value under byte, latency, false-positive, and uncertainty costs.

This contract joins the live predictor, residual stream, Sentinel evidence, reservoir geometry, multi-scale dynamics, HDC memory, codec, incident/discovery cards, and proof layer without claiming that statistical surprise alone contains semantics.

## 2. Strongest bounded claim

The strongest claim v1 is allowed to test is:

> On preregistered held-out streams and matched alert/byte budgets, a full-engine Eidos shadow policy can preserve decision-relevant deviations and replayable evidence at a better joint detection–memory–compression tradeoff than the strongest tested baseline, while a raw-residual escape channel prevents representation-induced recall loss.

This is an empirical conjecture, not an established theorem. It does not imply consciousness, universal understanding, causal diagnosis, production readiness, or superiority on every task.

## 3. Meaning requires a domain contract

For each domain \(d\), freeze a contract before evaluation:

\[
\mathcal{D}_d=(\mathcal{A}_d,Y^{(d)}_{t+1:t+H},\ell_d,H,c_d).
\]

- \(\mathcal{A}_d\): permitted actions, including `observe` or `no_alert`.
- \(Y^{(d)}_{t+1:t+H}\): a future outcome over a fixed horizon \(H\).
- \(\ell_d(a,Y)\): bounded loss for action \(a\) and outcome \(Y\).
- \(c_d\): fixed costs for bytes, latency, false alerts, and human review.

Labels may be used to score a completed run. They must not enter a representation, predictor, threshold, memory write, alert, codec decision, or card during the run.

If a domain has no frozen action/outcome/loss contract, Eidos may report `STRUCTURAL_HYPOTHESIS`; it may not report `OUTCOME_VALIDATED_MEANING`.

### 3.1 Identifiability boundary

No algorithm can infer universal semantic importance from an unlabeled observation distribution alone. Two worlds can emit the same stream while assigning different consequences to the same event. The stream is observationally identical but the optimal action differs. Therefore task-relative loss or later consequence evidence is necessary to validate meaning.

This boundary is a feature of the design: Eidos separates what it observed from what a domain says matters.

## 4. Stream and representation notation

Let:

- \(x_t\in\mathbb{R}^{d}\): current normalized input frame;
- \(h_t=(x_1,\ldots,x_{t-1})\): information available before frame \(t\);
- \(r_t\in\mathbb{R}^{N}\): live reservoir state;
- \(K\): number of candidate representation lifts;
- \(\Phi_k(h_t,x_t)=z_t^{(k)}\): causal representation lift \(k\);
- \(P_t^{(k)}(\cdot\mid h_t)\): prequential predictor in lift \(k\);
- \(\hat z_t^{(k)}\): its point prediction when required;
- \(R_t^{(k)}\): generalized residual;
- \(q\in\mathcal Q\): storage fidelity or codec mode;
- \(a\in\mathcal A_d\): bounded action or alert decision.

Every \(\Phi_k\) and \(P_t^{(k)}\) must use only information available by time \(t\). Test labels, future frames, test-wide means, and test-wide standard deviations are prohibited.

### 4.1 Required v1 lifts

The proof implementation must expose these candidates through a common interface:

| ID | Lift | Source |
| --- | --- | --- |
| `raw` | normalized input and live raw residual | live frame and `best_pred` |
| `spectral` | causal windowed spectral features | live stream window |
| `multiscale` | log-spaced causal time-scale features | fractal leak bands or an observer-equivalent filter bank |
| `geometry` | dominance, entropy, flatness, and state-change features | live reservoir state/eigen monitor |
| `memory` | HDC familiarity, bank, write state, and consequence recall | live hippocampus metrics |
| `consensus` | a calibrated combination of the above | proof-side representation scout |

The v1 scout may run in shadow mode. It must not replace the live engine prediction, Sentinel decision, HDC write/freeze policy, thermodynamic control, or codec output by default.

## 5. Generalized and quotient residuals

For vector-valued lifts, define a scale-normalized residual:

\[
R_t^{(k)}=\Omega_{k,t}^{-1/2}\left(z_t^{(k)}-\hat z_t^{(k)}\right),
\qquad
e_t^{(k)}=\rho_k\!\left(R_t^{(k)}\right),
\]

where \(\Omega_{k,t}\) is estimated from past calibration data only and \(\rho_k\) is a frozen robust norm or loss.

If a known nuisance group \(G_k\) acts on the representation, define the quotient residual:

\[
Q_t^{(k)}=\inf_{g\in G_k}
\rho_k\!\left(z_t^{(k)}-g\cdot\hat z_t^{(k)}\right).
\]

For a frozen nuisance subspace with projector \(P_{N_k}\), this becomes:

\[
Q_t^{(k)}=\left\|(I-P_{N_k})R_t^{(k)}\right\|_2.
\]

A quotient can remove nuisance variation, but it can also erase a meaningful event that lies inside the nuisance model. For that reason quotient evidence never replaces the raw channel.

## 6. Calibrated structural surprise

For each lift, maintain a calibration multiset \(\mathcal C_{k,t}\) containing only eligible past non-test residual scores. A finite-sample conformal-style tail value is:

\[
p_t^{(k)}=
\frac{1+\sum_{e\in\mathcal C_{k,t}}\mathbb 1[e\ge e_t^{(k)}]}
{1+|\mathcal C_{k,t}|}.
\]

The normalized structural evidence is:

\[
S_t^{(k)}=
\operatorname{clip}\left(
\frac{-\log p_t^{(k)}}{-\log p_{\min,k}},0,1
\right).
\]

The calibration method, eligibility rule, block/window length, \(p_{\min,k}\), and update cadence must be frozen per run. Under exchangeability the rank value has its usual finite-sample interpretation; under drift it is an operational score, not a guaranteed probability. Block calibration and held-out drift tests must make that limitation visible.

Projection norms are not assumed comparable. Every lift is calibrated separately. Any random projector must publish its scaling convention and a test for the claimed convention.

### 6.1 Persistence

Let \(b_t^{(k)}=\mathbb 1[S_t^{(k)}\ge\tau_k]\). Define decayed persistence:

\[
\Pi_t^{(k)}=(1-\eta_{\Pi})\Pi_{t-1}^{(k)}+\eta_{\Pi}b_t^{(k)}.
\]

An isolated spike and a sustained deviation can then have similar peak surprise but different persistence. The frozen domain contract decides whether persistence raises or lowers consequence.

## 7. Representation disagreement and phase invariant

When lift predictors emit probability distributions \(p_t^{(k)}(y)\), define normalized Jensen–Shannon disagreement:

\[
\Delta_t=
\frac{1}{\log K}
\sum_{k=1}^{K}\pi_{k,t}
\operatorname{KL}\!\left(p_t^{(k)}\middle\|\bar p_t\right),
\qquad
\bar p_t=\sum_k\pi_{k,t}p_t^{(k)}.
\]

If only scalar calibrated evidence is available, the explicitly named fallback is weighted evidence variance:

\[
\Delta_t^{\mathrm{var}}=
\frac{\sum_k\pi_{k,t}(S_t^{(k)}-\bar S_t)^2}{1/4},
\qquad
\bar S_t=\sum_k\pi_{k,t}S_t^{(k)},
\]

clipped to \([0,1]\). Results from the two definitions must not be pooled without identifying the definition.

For causal analytic phases \(\theta_{b,t}\) from \(B\) frozen time bands, define:

\[
C_{\phi,t}=\left|\sum_{b=1}^{B}w_b e^{i\theta_{b,t}}\right|,
\qquad \sum_bw_b=1.
\]

This coherence is invariant to a common phase rotation. Phase evidence must be derived from a causal filter or past-only window; centered filters that see the future are prohibited.

## 8. Familiarity is not danger

Let \(F_t\in[0,1]\) be HDC familiarity with a previously stored trace. Familiarity answers only: “does this resemble something stored?” It does not answer: “is this safe?”

Each eligible memory trace \(m\) may carry a delayed consequence value \(y_m\), its provenance, and confidence. With nonnegative similarity weights \(w_{m,t}\), define memory consequence risk:

\[
R_t^{\mathrm{mem}}=
\frac{\sum_{m\in\mathcal N_t}w_{m,t}y_m}
{\sum_{m\in\mathcal N_t}w_{m,t}},
\]

only when the effective support and confidence pass frozen minimums. Otherwise its state is `UNKNOWN`, not zero.

Required cases:

| Familiarity | Consequence recall | Permitted interpretation |
| --- | --- | --- |
| high | high risk | dangerous repeat; preserve/escalate even if novelty is low |
| high | low risk with adequate support | harmless familiar pattern; novelty may be reduced |
| high | unknown | familiar but unclassified; no safety de-escalation |
| low | any | novel or poorly recalled; use current evidence and uncertainty |

HDC familiarity may modulate learning plasticity as the current engine already does. It may not independently suppress a raw escape, a validated danger recall, or required evidence retention.

## 9. Value of information

For a frozen domain contract, define the risk before observing lift \(k\):

\[
\mathcal R_t^{0}=
\min_{a\in\mathcal A_d}
\mathbb E[\ell_d(a,Y_{t+1:t+H})\mid h_t].
\]

Define the risk after observing it:

\[
\mathcal R_t^{(k)}=
\min_{a\in\mathcal A_d}
\mathbb E[\ell_d(a,Y_{t+1:t+H})\mid h_t,z_t^{(k)}].
\]

Then:

\[
\operatorname{VOI}_t^{(k)}=\mathcal R_t^{0}-\mathcal R_t^{(k)}.
\]

True VOI is generally observed only after the horizon closes. Online control uses a prequential estimate \(\widehat{\operatorname{VOI}}_t^{(k)}\), trained on earlier outcomes only. Proof scoring uses realized held-out loss. The policy uses a conservative lower confidence bound:

\[
\underline V_t^{(k)}=
\widehat{\operatorname{VOI}}_t^{(k)}-\kappa_U\widehat\sigma_{V,t}^{(k)}.
\]

No result may label \(\widehat{\operatorname{VOI}}\) as realized value.

## 10. Meaningful Surprise decision object

Meaningful Surprise is a structured evidence object, not a single metaphysical number:

\[
\mathcal M_t^{(k)}=
\left(S_t^{(k)},Q_t^{(k)},\Pi_t^{(k)},\Delta_t,C_{\phi,t},F_t,
R_t^{\mathrm{mem}},\underline V_t^{(k)},U_t^{(k)}\right).
\]

The candidate set is:

\[
\mathcal K_t^+=
\left\{k:S_t^{(k)}\ge\tau_k
\;\lor\;\Pi_t^{(k)}\ge\tau_{\Pi}
\;\lor\;R_t^{\mathrm{mem}}\ge\tau_R\right\}
\cup\{\text{raw}\}.
\]

For candidate lift \(k\), fidelity \(q\), and action \(a\), define frozen net value:

\[
J_t(k,q,a)=
\underline V_t^{(k)}
+\lambda_R R_t^{\mathrm{mem}}
+\lambda_{\Pi}\Pi_t^{(k)}
+\lambda_{\Delta}\Delta_t
-\lambda_B\widetilde B_t(q)
-\lambda_{FP}\widehat C_{FP,t}(a)
-\lambda_L\widetilde L_t(a)
-\lambda_U U_t^{(k)}.
\]

All terms must be scaled using calibration data only. Coefficients, action set, fidelity set, and tie-breaking order are frozen before test evaluation. Familiarity \(F_t\) is intentionally absent as a negative risk term.

The shadow policy selects:

\[
(k_t^*,q_t^*,a_t^*)=
\arg\max_{k\in\mathcal K_t^+,q\in\mathcal Q,a\in\mathcal A_d}J_t(k,q,a)
\]

subject to the invariants below.

## 11. Safety and integrity invariants

### I1. Raw-residual escape

Let \(g_t^{\mathrm{raw}}=\mathbb 1[S_t^{\mathrm{raw}}\ge\tau_{\mathrm{raw,escape}}]\). When it fires:

- the event cannot be suppressed by quotient projection, representation selection, familiarity, or a negative VOI estimate;
- the raw residual and replay context must be retained at the frozen escape fidelity;
- the shadow action must be at least `review`;
- `raw_escape_triggered=true` must be visible in the card and receipt.

Budget pressure may be reported as a failure; it may not silently disable the escape.

### I2. Dangerous-repeat monotonicity

Holding current structural evidence fixed, increasing validated \(R_t^{\mathrm{mem}}\) may not reduce action urgency or storage fidelity.

### I3. Familiarity non-authority

Increasing \(F_t\) alone may reduce novelty or learning plasticity. It may not reduce validated consequence risk, override I1, or turn `UNKNOWN` consequence into safe.

### I4. Uncertainty monotonicity

When consequence risk is high, increasing uncertainty may increase evidence retention or request review; it may not justify discarding the evidence.

### I5. Outcome isolation

Future outcomes and truth labels are sealed from the online path. A test fails if any representation, threshold, action, memory update, or codec decision reads them.

### I6. Default-off compatibility

With `meaningful_surprise_enabled=false`, current outputs must be byte-for-byte identical for deterministic fixtures except explicitly allowed timestamp/path fields. The first implementation is a proof-side shadow layer.

### I7. Replay identity

Every decision carries source range, representation/config hash, code commit, seed, engine artifact references, and a replay command. Missing provenance prevents a proof verdict.

### I8. No semantic overclaim

Cards distinguish `STRUCTURAL_HYPOTHESIS`, `OUTCOME_ESTIMATED`, and `OUTCOME_VALIDATED`. An anomaly is not an attack, failure, seizure, or cause unless evidence outside the anomaly score establishes that statement.

## 12. Eidos Value reporting

The primary comparison is a value vector, not the current scalar in `eidos_brain.proof.run_proof`:

\[
\operatorname{EVV}=
(F_2,\operatorname{APR},\operatorname{EC},\operatorname{REP},\operatorname{MU},
-\operatorname{NBC},-\operatorname{NDL},-\operatorname{FPR},-U).
\]

- \(F_2\): event-level detection score emphasizing recall;
- APR: anomaly-preservation recall at the chosen byte budget;
- EC: evidence completeness under a common card wrapper;
- REP: deterministic replay success;
- MU: memory utility over harmless and dangerous repeats;
- NBC: compressed bytes divided by raw bytes;
- NDL: normalized detection delay;
- FPR: false positives per 10,000 nominal frames;
- \(U\): unresolved decision uncertainty.

Pareto comparison of EVV under matched budgets is primary. A scalar may be reported only as a preregistered secondary sensitivity analysis:

\[
\operatorname{EVM}_{\theta}=
\left[\prod_{j\in\{F_2,APR,EC,REP,MU\}}(\epsilon+j)^{w_j}\right]^{1/\sum_jw_j}
\exp[-(\lambda_BNBC+\lambda_LNDL+\lambda_{FP}FPR+\lambda_U U)].
\]

Weights, normalizers, \(\epsilon\), and penalty coefficients must be frozen. The claim must survive the registered sensitivity grid; one favorable weight choice is not evidence.

## 13. Output contract

Each shadow decision is a JSON object with at least:

```json
{
  "meaningful_surprise_version": "EIDOS-MS-v1",
  "frame_id": 0,
  "source_id": "...",
  "meaning_status": "STRUCTURAL_HYPOTHESIS",
  "domain_contract_hash": "sha256:...",
  "candidate_lifts": [],
  "selected_lift": "raw",
  "structural_evidence": 0.0,
  "quotient_residual": null,
  "persistence": 0.0,
  "representation_disagreement": 0.0,
  "phase_coherence": null,
  "familiarity": 0.0,
  "memory_consequence": {"state": "UNKNOWN", "risk": null},
  "voi": {"estimate": null, "lcb": null, "realized": null},
  "uncertainty": 1.0,
  "decision": {"action": "observe", "fidelity": "reference_or_null"},
  "safety": {"raw_escape_triggered": false, "overrides": []},
  "source_refs": [],
  "config_hash": "sha256:...",
  "code_commit": "...",
  "replay_command": "..."
}
```

Candidate lift rows must retain their own calibrated score, calibration status, prediction source, residual definition, and eligibility reason. The selected lift alone is insufficient for audit.

## 14. What can be proven by construction

1. **Raw-channel recall monotonicity.** If I1 is implemented as a final OR constraint, a raw escape cannot be removed by any learned lift or policy score.
2. **Common-phase invariance.** \(C_{\phi,t}\) is unchanged by \(\theta_{b,t}\mapsto\theta_{b,t}+\delta\) for all bands because the complex sum gains only a unit-magnitude factor \(e^{i\delta}\).
3. **Familiarity/danger separation.** If familiarity is excluded from the risk subtraction and I2–I3 are tested, high familiarity cannot by itself certify safety.
4. **Replay identity.** Given deterministic inputs, code/config hashes, seeds, and referenced source ranges, the proof layer can verify whether a decision is reproducible.

These construction properties do not prove that the chosen lifts discover useful structure. That requires the Grand Proof.

## 15. Falsification conditions

The v1 idea is weakened or rejected if any of the following occurs on held-out evaluation:

- the full-engine shadow policy does not improve the matched-budget Pareto frontier;
- the raw escape fails the nuisance-subspace counterexample;
- dangerous-repeat recall falls because familiarity suppresses evidence;
- gains disappear under common evidence/replay wrappers for baselines;
- gains disappear under ablation or are attributable only to a larger alert/byte budget;
- VOI estimates do not predict realized loss reduction out of sample;
- cross-domain performance collapses beyond the registered margin;
- artifact identity, label isolation, or replay checks fail.

A negative result is useful: it identifies whether the missing mechanism is representation, consequence estimation, memory policy, calibration, or resource allocation.

## 16. Smallest safe implementation path

1. Add a proof-side frame observer that consumes existing live engine fields: `best_pred`, residuals, Sentinel metrics, geometry, HDC metrics, thermodynamic state, codec decision, and replay metadata.
2. Add representation lifts and calibrators outside the core engine.
3. Emit shadow Meaningful Surprise decisions and shadow codec tokens into a separate artifact tree.
4. Keep the feature disabled by default and prove baseline-output invariance.
5. Run the Grand Proof and ablations.
6. Only after a passing result, propose a separate promotion protocol for allowing the policy to control production alerts or codec fidelity.

Leap IV / multiverse execution, a larger reservoir, and new physics controllers are outside v1. They would add degrees of freedom before the existing architecture has closed its central proof loop.

## 17. Repository integration seams

The implementation should reuse rather than duplicate:

- live entrypoint: `repo/src/eidos_brain/engine/adapters.py::run_session`;
- live signals and codec metadata: `repo/src/eidos_brain/engine/eidos_v0_4_7_02.py::run_sentinel_stream`;
- codec: `repo/src/eidos_brain/compression/residual_codec.py`;
- codec policy: `repo/src/eidos_brain/compression/policy.py`;
- hidden-structure proof package: `repo/src/eidos_brain/proof/`;
- labeled runner and its raw/merged/deduped/confirmed/calibrated views: `tools/run_labeled_domain_proof.py`;
- controlled streams: `repo/src/eidos_brain/experiments/controlled_regimes.py`;
- incident-card explanation contract: `docs/proof/incident_card_operator_explanation_v1.md`.

The existing `eidos_minimal` and quantum benchmark Sentinel proxy are baselines or smoke tools. They are not substitutes for a full-engine proof.

## 18. Proof Logic + Meaning

**Goal.** Turn Eidos from a collection of strong mechanisms into one falsifiable system that preserves what changes a bounded decision.

**Logic and math.** Calibrated residual evidence answers “how unexpected?”; persistence answers “how sustained?”; disagreement answers “do representations conflict?”; HDC answers “have we seen this?”; memory consequence answers “what happened last time?”; VOI answers “does observing this change expected loss?”; the constrained policy answers “what should we preserve or surface at this cost?”

**Philosophical meaning.** Meaning is not hidden inside a z-score. It is a relationship among an observation, a remembered context, a possible future, and an action. Eidos becomes more powerful by making that relationship explicit and auditable.

**Why this is better.** The current engine can detect, remember, monitor geometry, regulate itself, compress, and explain, but those channels do not share one consequence-aware objective. This contract gives them a common decision boundary without erasing their separate meanings.

**Movement toward the north star.** If the Grand Proof passes, Eidos can make a bounded and genuinely distinctive claim: it preserves decision-relevant anomalies with replayable evidence under resource constraints, rather than merely producing another anomaly score or another compressor.

**Evidence still required.** Full-engine matched-budget experiments, ablations, cross-domain held-out evaluation, deterministic replay, and independent operator review. Until those exist, this document is a design and falsification contract—not proof of advantage.
