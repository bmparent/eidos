export class RunState {
  constructor() {
    this.runId = `run-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
    this.runEpoch = 0;
    this.resetCount = 0;
    this.totalGenerations = 0;
    this.lastObservedGeneration = 0;
    this.startedAt = new Date().toISOString();
    this.lastResetAt = null;
    this.resetEvents = [];
  }

  updateGeneration(currentGeneration, context = {}) {
    const prev = this.lastObservedGeneration;
    if (currentGeneration < prev) this.recordReset('detected_generation_drop', prev, currentGeneration, context);
    const delta = Math.max(0, currentGeneration - this.lastObservedGeneration);
    this.totalGenerations += delta;
    this.lastObservedGeneration = currentGeneration;
  }

  recordReset(resetReason, previousGeneration, currentGeneration, context = {}) {
    this.runEpoch += 1;
    this.resetCount += 1;
    this.lastResetAt = new Date().toISOString();
    this.resetEvents.push({ resetReason, previousGeneration, currentGeneration, timestamp: this.lastResetAt, scenario: context.scenario || null, settings: context.settings || null });
    if (this.resetEvents.length > 200) this.resetEvents.shift();
  }

  exportMeta() {
    return { runId:this.runId, runEpoch:this.runEpoch, resetCount:this.resetCount, totalGenerations:this.totalGenerations, lastObservedGeneration:this.lastObservedGeneration, startedAt:this.startedAt, lastResetAt:this.lastResetAt, resetEvents:this.resetEvents.slice(-50) };
  }
}
