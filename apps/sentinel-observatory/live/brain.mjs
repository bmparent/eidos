/**
 * Portable numerical port of Eidos Sentinel V3's one-feature, 32-reservoir profile.
 * This IS a port, not the unchanged Python runtime. Constants and parity vectors
 * are generated from the canonical NumPy implementation. No network or Node APIs.
 */
// BEGIN GENERATED BRAIN CONSTANTS
const BRAIN_CONSTANTS = Object.freeze({
  "wIn": [
    0.6412773650839229,
    -0.2053586009546613,
    -0.5758999671329448,
    1.186616672368152,
    -1.8273831546840644,
    0.3015707363505245,
    1.36581833106114,
    -2.5011617584652996,
    0.2675554791850724,
    -0.0066704941385530275,
    -0.2943461731751522,
    -0.2490663372867569,
    -0.06481091822304745,
    1.769338399177576,
    1.8541888034274472,
    0.41052427724011503,
    -0.5835209684828736,
    0.806888968857361,
    -0.42445035446213975,
    1.2018287698552228,
    -0.7659183414869153,
    -0.6406122952434946,
    0.28076465688988483,
    -1.3699546257238946,
    -0.3071265274745126,
    0.6183710955397933,
    -0.7637129153274229,
    -0.8589979039028967,
    0.0639350221117633,
    -0.2731819683266816,
    0.28735045590321273,
    -0.0058253944216024416
  ],
  "recurrentSigns": [
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    -1.0
  ],
  "probe": -1.0,
  "sourceHashes": {
    "sentinel_v3.py": "b76207726fbcc29f626bdef773519c13f26885c2839c34825dee565906a1eef5",
    "calibration.py": "f2f07b148c17f2734fd089fd02614231bf82bb7e3789c1643636fec7c9d64505",
    "input_detectors.py": "7bee75283675893bf43b5e64551c6cd640adec0859f4c4361b075b85dd631253",
    "hysteresis.py": "f63785d58ca85a21fe2abd734b335a5d064d832f51b762466f97c7fda2390b66",
    "normal_suppression.py": "2b1c820a299335e863a1c2451fbbe000a68215f3d53563a2d08f01e39467cc99",
    "event_merge.py": "50ace015d4bc7ad4577b8bc41f2e28a390b2292d8f47303aa1b962bd86e46f9d"
  },
  "config": {
    "features": 1,
    "reservoir_size": 32,
    "warmup": 12,
    "rolling_window": 8,
    "seed": 42,
    "periodic_probe": "disabled"
  }
});
// END GENERATED BRAIN CONSTANTS

export const ENGINE_PROVENANCE = Object.freeze({
  engine: 'Eidos Brain / Sentinel V3 · portable JS port',
  language: 'JavaScript binary64', profile: BRAIN_CONSTANTS.config,
  canonical_source_hashes: BRAIN_CONSTANTS.sourceHashes,
  validation: 'Numerical parity tested against canonical Python; not security validation',
  scope: 'Current-frame log-latency behavior; not threat classification',
});

const finite = (x, fallback = 0) => Number.isFinite(x) ? x : fallback;
const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const mean = a => a.length ? a.reduce((sum, x) => sum + x, 0) / a.length : 0;
const norm = a => Math.sqrt(a.reduce((sum, x) => sum + x * x, 0));
const variance = a => { const m = mean(a); return a.length ? a.reduce((sum, x) => sum + (x - m) ** 2, 0) / a.length : 0; };
const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const scaled = (x, threshold = 3, scale = 4) => x <= threshold ? 0 : Math.min(3, (x - threshold) / scale);
const stamp = x => typeof x === 'string' && /(?:Z|[+-]\d\d:\d\d)$/i.test(x) && Number.isFinite(Date.parse(x)) ? Date.parse(x) : NaN;
const boundedLatency = x => typeof x === 'number' && Number.isFinite(x) && x >= 0 && x <= 86400000;

class Stats {
  constructor() { this.count = 0; this.mean = 0; this.m2 = 0; }
  update(x) { this.count++; const d = x - this.mean; this.mean += d / this.count; this.m2 += d * (x - this.mean); }
  z(x) { const v = this.count < 2 ? 1e-9 : Math.max(1e-9, this.m2 / (this.count - 1)); return clip((x - this.mean) / Math.sqrt(v), -99, 99); }
}

/** Low-level canonical port. Exported for numeric parity, not a security verdict. */
export class SentinelPort {
  constructor() {
    this.state = Array(32).fill(0); this.wOut = Array(32).fill(0); this.pDiag = Array(32).fill(100);
    this.t = 0; this.ema = 0; this.residuals = []; this.previous = null; this.rolling = [];
    this.stats = Object.fromEntries(['energy', 'delta', 'rolling', 'signature', 'plasticity', 'stateVariance', 'dominance'].map(k => [k, new Stats()]));
    this.eventScore = 0; this.multiplier = 1; this.collapse = 0; this.freeze = 0;
    this.episode = 0; this.activeEpisode = 0; this.episodeAge = 0; this.cooldown = 0; this.recoveries = 0;
  }
  sigma() {
    if (this.residuals.length < 8) return 1e-6;
    const med = median(this.residuals), mad = median(this.residuals.map(x => Math.abs(x - med)));
    return Math.max(1e-6, mad > 0 ? 1.4826 * mad : Math.sqrt(variance(this.residuals)));
  }
  step(x) {
    if (typeof x !== 'number' || !Number.isFinite(x) || Math.abs(x) > 1e6) throw new Error('one bounded finite numeric feature required');
    const t = this.t, warmup = t < 12, prev = [...this.state];
    this.state = prev.map((v, i) => .65 * v + .35 * Math.tanh(BRAIN_CONSTANTS.wIn[i] * x + .18 * prev[(i + 31) % 32] * BRAIN_CONSTANTS.recurrentSigns[i]));
    const predicted = this.wOut.reduce((sum, w, i) => sum + w * this.state[i], 0);
    const err = x - predicted, errorNorm = Math.abs(err);
    const z = Math.abs(errorNorm - (t ? this.ema : errorNorm)) / this.sigma();
    this.ema = t ? .97 * this.ema + .03 * errorNorm : errorNorm;
    this.residuals.push(errorNorm); if (this.residuals.length > 64) this.residuals.shift();
    const energy = Math.abs(x), delta = this.previous === null ? 0 : Math.abs(x - this.previous);
    this.rolling.push(x); if (this.rolling.length > 8) this.rolling.shift();
    const rollingVar = this.rolling.length < 2 ? 0 : variance(this.rolling), signature = x * BRAIN_CONSTANTS.probe;
    if (warmup) { this.stats.energy.update(energy); this.stats.delta.update(delta); this.stats.rolling.update(rollingVar); this.stats.signature.update(signature); }
    const energyZ = this.stats.energy.z(energy), deltaZ = this.stats.delta.z(delta), rollingZ = this.stats.rolling.z(rollingVar), signatureZ = this.stats.signature.z(signature);
    const inputRaw = Math.max(scaled(Math.abs(energyZ)), scaled(Math.abs(deltaZ)), scaled(Math.max(0, rollingZ)), scaled(Math.max(0, -rollingZ), 2.25));
    const noveltyRaw = scaled(Math.abs(signatureZ), 3.5);
    this.previous = x;
    const plasticity = norm(this.state.map((value, i) => value - prev[i])) / Math.sqrt(32);
    const stateVar = variance(this.state), stateNorm = norm(this.state);
    const dominance = Math.max(...this.state.map(Math.abs)) / (stateNorm / Math.sqrt(32) + 1e-9);
    if (warmup) { this.stats.plasticity.update(plasticity); this.stats.stateVariance.update(stateVar); this.stats.dominance.update(dominance); }
    const plasticityZ = this.stats.plasticity.z(plasticity), stateVarZ = this.stats.stateVariance.z(stateVar), dominanceZ = this.stats.dominance.z(dominance);
    const lowPlasticity = scaled(Math.max(0, -plasticityZ), 2, 3), lowStateVar = scaled(Math.max(0, -stateVarZ), 2, 3);
    const rankCollapse = Math.max(lowPlasticity, lowStateVar, scaled(Math.max(0, dominanceZ), 3, 4));
    const geometryRaw = Math.max(rankCollapse, Math.min(3, .7 * scaled(Math.max(0, -rollingZ), 2, 3) + .5 * lowPlasticity));
    const residualEvidence = warmup ? 0 : scaled(z, 3.5, 4), inputEvidence = warmup ? 0 : inputRaw;
    const geometryEvidence = warmup ? 0 : geometryRaw, noveltyEvidence = warmup ? 0 : noveltyRaw;
    this.multiplier = Math.max(residualEvidence, inputEvidence, geometryEvidence) <= .12 ? Math.min(1.35, this.multiplier + .015) : Math.max(1, this.multiplier - .08);
    const total = .35 * residualEvidence + 1.15 * inputEvidence + 1.35 * geometryEvidence + .25 * noveltyEvidence;
    this.eventScore = finite(.9 * this.eventScore + .58 * (warmup ? 0 : total));
    this.collapse = geometryEvidence >= .35 && !warmup ? this.collapse + 1 : 0;
    const status = warmup ? 'GREEN' : this.collapse >= 18 ? 'RED' : this.eventScore >= this.multiplier ? 'AMBER' : 'GREEN';
    if (status === 'RED') this.freeze = Math.max(this.freeze, 48);
    const frozen = !warmup && this.freeze > 0, scale = frozen ? 0 : status === 'AMBER' ? .25 : 1;
    const safety = this.adapt(err, scale);
    if (this.freeze > 0 && status !== 'RED') this.freeze--;
    const merge = this.merge(status);
    this.t++;
    // Canonical result.update(geometry_fields) exposes raw geometry during warmup;
    // the wrapper suppresses warmup signals while the low-level port preserves it.
    return {t, status, is_surprise: residualEvidence > 0, residual_evidence: residualEvidence,
      input_evidence: inputEvidence, geometry_evidence: geometryRaw, novelty_evidence: noveltyRaw,
      event_score: this.eventScore, adaptation_frozen: frozen, ema_err: this.ema, sigma: this.sigma(), z,
      error_norm: errorNorm, threshold_multiplier: this.multiplier, collapse_counter: this.collapse, freeze_remaining: this.freeze, warmup,
      energy, delta, rolling_var: rollingVar, period_power: 0, energy_z: energyZ, delta_z: deltaZ, rolling_var_z: rollingZ,
      period47_score: 0, input_evidence_score: inputRaw, plasticity, state_variance: stateVar, eigen_dominance: dominance,
      plasticity_z: plasticityZ, state_var_z: stateVarZ, eigen_dominance_z: dominanceZ,
      low_plasticity_score: lowPlasticity, low_state_var_score: lowStateVar, rank_collapse_score: rankCollapse,
      ...safety, ...merge};
  }
  adapt(err, scale) {
    if (scale <= 0) return {rls_recovered: false, rls_p_clipped: false, w_out_clipped: false, recoveries: this.recoveries};
    const pr = this.pDiag.map((p, i) => p * this.state[i]);
    const denom = .995 + this.state.reduce((sum, value, i) => sum + value * pr[i], 0);
    if (!Number.isFinite(denom) || denom <= 1e-12 || ![...this.state, ...this.wOut, ...this.pDiag].every(Number.isFinite)) {
      this.recoveries++; this.state = this.state.map(x => finite(x)); this.wOut = this.wOut.map(x => clip(finite(x), -6, 6)); this.pDiag = Array(32).fill(100);
      return {rls_recovered: true, rls_p_clipped: true, w_out_clipped: true, recoveries: this.recoveries};
    }
    const gain = pr.map(p => p / denom);
    const nextW = this.wOut.map((w, i) => w + scale * err * gain[i]);
    const nextP = this.pDiag.map((p, i) => (p - gain[i] * pr[i]) / .995);
    this.pDiag = nextP.map(x => clip(finite(x, 100), 1e-6, 1e4));
    this.wOut = nextW.map(x => clip(finite(x), -6, 6));
    const wNorm = norm(this.wOut); if (wNorm > 250) this.wOut = this.wOut.map(x => x * (250 / wNorm));
    return {rls_recovered: false, rls_p_clipped: nextP.some((p, i) => p !== this.pDiag[i]), w_out_clipped: nextW.some((w, i) => w !== this.wOut[i]), recoveries: this.recoveries};
  }
  merge(status) {
    if (status === 'AMBER' || status === 'RED') {
      if (this.cooldown <= 0) { this.episode++; this.activeEpisode = this.episode; this.episodeAge = 0; }
      this.cooldown = 24; this.episodeAge++;
    } else if (this.cooldown > 0) { this.cooldown--; this.episodeAge++; }
    else { this.activeEpisode = 0; this.episodeAge = 0; }
    return {event_active: this.activeEpisode !== 0, episode_id: this.activeEpisode, episode_age: this.episodeAge, merge_cooldown_remaining: this.cooldown};
  }
}

const signalDefinitions = [
  ['residual_evidence', 'Reconstruction surprise', 'Current-frame residual differs from recent residuals; not a future prediction'],
  ['input_evidence', 'Latency distribution change', 'Latency energy, change or variation differs from provisional calibration'],
  ['geometry_evidence', 'Reservoir geometry change', 'Internal state variation changed; not a security severity'],
  ['novelty_evidence', 'Input novelty', 'Projected latency differs from provisional calibration'],
];

export class LiveBrain {
  constructor({epochId = `epoch-${Date.now()}`, startedAt = new Date().toISOString(), now = () => new Date().toISOString()} = {}) {
    if (!Number.isFinite(stamp(startedAt))) throw new Error('startedAt requires a timezone-aware timestamp');
    this.epochId = String(epochId).slice(0, 200); this.startedAt = startedAt; this.sources = new Map(); this.now = now;
  }
  newSource() { return {detector: new SentinelPort(), samples: 0, rejected: 0, last: null, lastId: null, ewma: null,
    pending: null, scored: 0, error: 0, baselineError: 0, squaredError: 0, baselineSquaredError: 0,
    wins: 0, baselineWins: 0, ties: 0, skipped: 0, seen: new Set(), state: 'calibrating', gap: null}; }
  comparison(s) { return {scored_count: s.scored, minimum_provisional_comparison: 20,
    status: s.scored >= 20 ? 'provisional_observed_window' : 'insufficient_evidence',
    ewma_mae_ms: s.scored ? s.error / s.scored : null, persistence_mae_ms: s.scored ? s.baselineError / s.scored : null,
    ewma_rmse_ms: s.scored ? Math.sqrt(s.squaredError / s.scored) : null,
    persistence_rmse_ms: s.scored ? Math.sqrt(s.baselineSquaredError / s.scored) : null,
    paired_mae_improvement_ms: s.scored ? (s.baselineError - s.error) / s.scored : null,
    ewma_wins: s.wins, persistence_wins: s.baselineWins, ties: s.ties, skipped_forecasts: s.skipped,
    skill_established: false, limitation: 'Sequential probe latency only; short duration and proxy variation prevent general skill claims'}; }
  observe(event) {
    const source = event?.source_id, id = event?.id, completed = event?.completed_at, started = event?.observed_at;
    if (typeof source !== 'string' || !source || source.length > 200) throw new Error('valid source_id required');
    if (!this.sources.has(source)) {
      if (this.sources.size >= 32) throw new Error('source limit reached; start a new observation epoch');
      this.sources.set(source, this.newSource());
    }
    const s = this.sources.get(source), time = stamp(completed), requestTime = stamp(started);
    const clientStart = stamp(event.client_request_started_at), clientReceipt = stamp(event.client_received_at), issuedNow = this.now();
    const clientNow = stamp(issuedNow);
    const base = {source_id: source, evidence_id: id, engine: ENGINE_PROVENANCE.engine, provenance: ENGINE_PROVENANCE,
      session_epoch: this.epochId, session_started_at: this.startedAt, observed_status: event.status ?? null,
      engine_scope: ENGINE_PROVENANCE.scope, calibration: 'provisional 12-response baseline; no labeled security validation',
      observation_timestamp_scope: 'server response_completed_at',
      forecast_timestamp_scope: 'Browser clock at actual computation; compared only with next browser request start',
      clock_domains: 'Server response timing and browser forecast issuance are distinct; no cross-clock ordering claim'};
    let reason = null;
    if (typeof id !== 'string' || !id || id.length > 256) reason = 'valid evidence id required';
    else if (s.seen.has(id)) reason = 'duplicate observation excluded';
    else if (!Number.isFinite(time) || !Number.isFinite(requestTime) || requestTime > time) reason = 'invalid request/response timestamps';
    else if (![clientStart, clientReceipt, clientNow].every(Number.isFinite) || clientStart > clientReceipt || clientReceipt > clientNow) reason = 'missing or invalid browser timing provenance';
    else if (clientStart < stamp(this.startedAt)) reason = 'historical observation excluded from current live epoch';
    else if (s.last !== null && time <= s.last) reason = 'non-increasing response timestamp excluded';
    else if (!Number.isInteger(event.status) || event.status < 100 || event.status > 599 || event.error != null || event.response_chain_complete === false || !boundedLatency(event.latency_ms)) reason = 'missing or incomplete HTTP response or invalid response latency excluded';
    if (reason) {
      s.rejected++;
      if (reason.startsWith('missing') && s.pending) { s.skipped++; s.pending = null; }
      return {...base, state: 'insufficient_data', reason, warmup_remaining: Math.max(0, 12 - s.samples), event_score: null, signals: [],
        forecast: {issued: null, scored: null, comparison: this.comparison(s)}, missing_data_policy: 'Excluded; never replaced with zero'};
    }
    const latency = event.latency_ms, gap = s.last === null ? null : (time - s.last) / 1000;
    s.gap = gap !== null && gap > 300 ? {seconds: gap, note: 'Long coverage gap; prior calibration retained and remains provisional'} : null;
    let scored = null;
    if (s.pending) {
      const frozen = s.pending, horizon = (clientReceipt - stamp(frozen.issued_at)) / 1000;
      if (stamp(frozen.issued_at) < clientStart && horizon > 0 && horizon <= 300) {
        const ae = Math.abs(frozen.predicted_latency_ms - latency), be = Math.abs(frozen.persistence_latency_ms - latency);
        s.scored++; s.error += ae; s.baselineError += be; s.squaredError += ae * ae; s.baselineSquaredError += be * be;
        if (Math.abs(ae - be) < 1e-9) s.ties++; else if (ae < be) s.wins++; else s.baselineWins++;
        scored = {...frozen, status: 'scored', actual_evidence_id: id, observed_at: completed,
          client_request_started_at: event.client_request_started_at, client_received_at: event.client_received_at, actual_latency_ms: latency,
          absolute_error_ms: ae, persistence_error_ms: be, horizon_seconds: horizon, issued_before_request: true,
          timing_proof: 'Prior browser computation precedes this browser request on the same clock'};
      } else { s.skipped++; scored = {forecast_id: frozen.forecast_id, status: 'unscored', reason: 'Late or overlapping request, or forecast horizon exceeded', horizon_seconds: horizon}; }
    }
    const row = s.detector.step(Math.log1p(latency)); s.samples++; s.last = time; s.lastId = id;
    s.seen.add(id); if (s.seen.size > 2048) s.seen.delete(s.seen.values().next().value);
    s.ewma = s.ewma === null ? latency : .35 * latency + .65 * s.ewma;
    s.state = row.warmup ? 'calibrating' : row.status === 'RED' ? 'geometry_change' : row.status === 'AMBER' ? 'change_detected' : 'observing';
    // Timestamp publication after computing this observation's next forecast.
    const publishedAt = this.now();
    s.pending = s.samples >= 12 ? {forecast_id: `latency-next:${this.epochId}:${source}:${id}`, method: 'EWMA latency', alpha: .35,
      target: 'next observed HTTP response latency for this source', issued_at: publishedAt,
      basis_observed_at: completed, basis_client_received_at: event.client_received_at, basis_evidence_id: id,
      issuance_clock: 'browser',
      predicted_latency_ms: s.ewma, persistence_latency_ms: latency, max_horizon_seconds: 300, status: 'pending',
      claim: 'experimental numerical forecast, not incident prediction'} : null;
    return {...base, state: s.state, sample_count: s.samples, warmup_remaining: Math.max(0, 12 - s.samples),
      event_score: row.warmup ? null : row.event_score, episode_id: row.episode_id, event_active: !row.warmup && row.event_active,
      coverage_gap: s.gap, signals: row.warmup ? [] : signalDefinitions.map(([key, label, meaning]) => ({key, label, meaning, value: row[key], evidence_ids: [id]})),
      diagnostics: {reconstruction_error: row.error_norm, reconstruction_z: row.z, feature: 'log1p(latency_ms)', periodic_probe: 'disabled', security_verdict: 'not assessed'},
      forecast: {issued: s.pending ? {...s.pending} : null, scored, comparison: this.comparison(s)}};
  }
  summary() { return {engine: ENGINE_PROVENANCE.engine, provenance: ENGINE_PROVENANCE, session_epoch: this.epochId,
    session_started_at: this.startedAt, restart_policy: 'New live calibration per browser epoch; no historical backfill',
    sources: [...this.sources].map(([source_id, s]) => ({source_id, sample_count: s.samples, rejected_count: s.rejected,
      state: s.state, warmup_remaining: Math.max(0, 12 - s.samples), last_evidence_id: s.lastId,
      last_observed_at: s.last === null ? null : new Date(s.last).toISOString(), coverage_gap: s.gap, forecast: this.comparison(s)}))}; }
}

/** Evidence-bound narration only. No model invocation, no root-cause assertion. */
export function analyzeEvidence(event, recent = []) {
  const brain = event?.brain, ids = [event.id], status = event.status;
  const facts = [{text: status == null ? 'The probe completed without a usable HTTP response at this vantage.' :
    `A real HEAD probe received HTTP ${status} in ${Number(event.latency_ms).toFixed(1)} ms.`, evidence_ids: ids}];
  if (brain?.state === 'change_detected' || brain?.state === 'geometry_change') facts.push({text: 'The experimental Sentinel latency signal differs from its provisional baseline.', evidence_ids: ids});
  const prior = recent.filter(item => item && item.source_id === event.source_id && item.id !== event.id && stamp(item.completed_at) < stamp(event.completed_at)).slice(-4);
  if (prior.length) facts.push({text: `There are ${prior.length} preceding observations of this same target available for comparison.`, evidence_ids: prior.map(item => item.id)});
  return {kind: 'deterministic', model_invoked: false, evidence_ids: [...ids, ...prior.map(item => item.id)],
    summary: facts[0].text, facts, hypotheses: status == null ? [
      {text: 'Network, proxy, DNS or destination delay are competing explanations; this probe cannot distinguish them.', evidence_ids: ids, status: 'unresolved'}
    ] : [], next_observations: ['Compare the next samples and an independent authorized vantage before attributing cause.'],
    unknowns: ['Visitor traffic, origin logs and internal network were not observed.', 'This does not establish an attack, outage, or predictive skill.']};
}
