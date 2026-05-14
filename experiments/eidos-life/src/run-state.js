const STORAGE_KEYS = {
  activeMeta: 'eidos-life:active-run-meta',
  lastObservedGeneration: 'eidos-life:last-observed-generation',
  lastHeartbeat: 'eidos-life:last-run-heartbeat',
  resetHistory: 'eidos-life:reset-history',
};

const RESET_DROP_THRESHOLD = 25;

export class RunState {
  constructor({ storage = globalThis.localStorage, now = () => new Date().toISOString(), pageLoadId = null } = {}) {
    this.storage = storage;
    this.now = now;
    this.pageLoadId = pageLoadId || `page-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    this.runId = `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    this.runEpoch = 0;
    this.resetCount = 0;
    this.totalGenerations = 0;
    this.lastObservedGeneration = 0;
    this.highestObservedGeneration = 0;
    this.startedAt = this.now();
    this.lastResetAt = null;
    this.lastHeartbeatAt = null;
    this.lastExportAt = null;
    this.resetEvents = [];
    this.reloadCount = 0;
    this.continuityStatus = 'active';
    this.diagnostics = { continuityRestored: false, durableResetDetected: false, hiddenIntervals: [], longGaps: [] };
  }

  initialize({ currentGeneration = 0, scenario = null, settingsHash = null } = {}) {
    const prior = this.#readJson(STORAGE_KEYS.activeMeta, null);
    this.lastObservedGeneration = currentGeneration;
    this.highestObservedGeneration = currentGeneration;
    if (!prior) {
      this.persistMeta({ scenario, settingsHash });
      return;
    }
    this.runId = prior.runId || this.runId;
    this.runEpoch = prior.runEpoch || 0;
    this.resetCount = prior.resetCount || 0;
    this.totalGenerations = Math.max(prior.totalGenerations || 0, prior.highestObservedGeneration || 0);
    this.lastObservedGeneration = currentGeneration;
    this.highestObservedGeneration = Math.max(prior.highestObservedGeneration || 0, currentGeneration);
    this.startedAt = prior.startedAt || this.startedAt;
    this.resetEvents = Array.isArray(prior.resetEvents) ? prior.resetEvents.slice(-200) : [];
    this.reloadCount = (prior.reloadCount || 0) + 1;
    this.diagnostics.continuityRestored = true;

    const previousGen = Math.max(prior.highestObservedGeneration || 0, prior.lastObservedGeneration || 0);
    if (previousGen - currentGeneration >= RESET_DROP_THRESHOLD) {
      this.recordReset('app_restart_generation_drop', previousGen, currentGeneration, {
        previousRunId: prior.runId,
        previousTotalGenerations: prior.totalGenerations || 0,
        newPageLoadId: this.pageLoadId,
        durable: true,
      });
      this.totalGenerations = Math.max(this.totalGenerations, previousGen);
      this.continuityStatus = `restart_detected_${previousGen}_to_${currentGeneration}`;
      this.diagnostics.durableResetDetected = true;
    } else {
      this.continuityStatus = 'restored';
    }
    this.persistMeta({ scenario: scenario || prior.scenario || null, settingsHash: settingsHash || prior.settingsHash || null });
  }

  updateGeneration(currentGeneration, context = {}) {
    const prev = this.lastObservedGeneration;
    if (currentGeneration < prev) this.recordReset('detected_generation_drop', prev, currentGeneration, context);
    const delta = Math.max(0, currentGeneration - prev);
    this.totalGenerations += delta;
    this.lastObservedGeneration = currentGeneration;
    this.highestObservedGeneration = Math.max(this.highestObservedGeneration, currentGeneration);
  }

  recordReset(resetReason, previousGeneration, currentGeneration, context = {}) {
    this.runEpoch += 1;
    this.resetCount += 1;
    this.lastResetAt = this.now();
    this.continuityStatus = 'reset_detected';
    const evt = { resetReason, previousGeneration, currentGeneration, timestamp: this.lastResetAt, scenario: context.scenario || null, settings: context.settings || null, previousRunId: context.previousRunId || null, previousTotalGenerations: context.previousTotalGenerations || null, newPageLoadId: context.newPageLoadId || this.pageLoadId };
    this.resetEvents.push(evt);
    if (this.resetEvents.length > 200) this.resetEvents.shift();
    this.#writeJson(STORAGE_KEYS.resetHistory, this.resetEvents.slice(-200));
  }

  noteVisibilityChange(state) {
    const at = this.now();
    if (state === 'hidden') this._hiddenAt = at;
    if (state === 'visible' && this._hiddenAt) {
      const gapMs = new Date(at).getTime() - new Date(this._hiddenAt).getTime();
      this.diagnostics.hiddenIntervals.push({ hiddenAt: this._hiddenAt, visibleAt: at, gapMs });
      if (gapMs > 30000) this.diagnostics.longGaps.push({ hiddenAt: this._hiddenAt, visibleAt: at, gapMs });
      this._hiddenAt = null;
    }
  }

  markExported() { this.lastExportAt = this.now(); }

  heartbeat({ scenario = null, settingsHash = null, visibilityState = 'visible' } = {}) {
    this.lastHeartbeatAt = this.now();
    this.persistMeta({ scenario, settingsHash, visibilityState });
  }

  persistMeta({ scenario = null, settingsHash = null, visibilityState = null } = {}) {
    const meta = this.exportMeta();
    meta.scenario = scenario;
    meta.settingsHash = settingsHash;
    meta.visibilityState = visibilityState;
    this.#writeJson(STORAGE_KEYS.activeMeta, meta);
    this.#writeJson(STORAGE_KEYS.lastObservedGeneration, { generation: this.lastObservedGeneration, at: this.now() });
    this.#writeJson(STORAGE_KEYS.lastHeartbeat, { at: this.lastHeartbeatAt || this.now(), pageLoadId: this.pageLoadId });
  }

  clearDurableState() {
    Object.values(STORAGE_KEYS).forEach((key) => this.storage?.removeItem?.(key));
  }

  exportMeta() {
    return { runId: this.runId, runEpoch: this.runEpoch, resetCount: this.resetCount, totalGenerations: this.totalGenerations, lastObservedGeneration: this.lastObservedGeneration, highestObservedGeneration: this.highestObservedGeneration, startedAt: this.startedAt, lastResetAt: this.lastResetAt, lastHeartbeatAt: this.lastHeartbeatAt, lastExportAt: this.lastExportAt, pageLoadId: this.pageLoadId, reloadCount: this.reloadCount, resetEvents: this.resetEvents.slice(-50), continuityStatus: this.continuityStatus, diagnostics: { ...this.diagnostics, hiddenIntervals: this.diagnostics.hiddenIntervals.slice(-10), longGaps: this.diagnostics.longGaps.slice(-10) } };
  }

  #readJson(key, fallback) { try { const raw = this.storage?.getItem?.(key); return raw ? JSON.parse(raw) : fallback; } catch { return fallback; } }
  #writeJson(key, value) { try { this.storage?.setItem?.(key, JSON.stringify(value)); } catch {} }
}
