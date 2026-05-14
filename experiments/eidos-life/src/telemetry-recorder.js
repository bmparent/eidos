export class TelemetryRecorder {
  constructor(options = {}) {
    this.startedAt = new Date().toISOString();
    this.maxRowsFull = options.maxRowsFull ?? 20000;
    this.sampleEveryAfterMax = options.sampleEveryAfterMax ?? 10;
    this.maxSampledRows = options.maxSampledRows ?? 20000;
    this.maxEvents = options.maxEvents ?? 800;
    this.maxRegimeTransitions = options.maxRegimeTransitions ?? 2000;
    this.rows = [];
    this.sampledRows = [];
    this.totalRowsRecorded = 0;
    this.events = [];
    this.recentEvents = [];
    this.prevRegime = 'CALIBRATING';
    this.largestMass = 0;
    this.evolution = { genomes: [], lineages: [], organisms: [], deadOrganisms: [], lineageGraph: [], events: [] };
    this.regimeCounts = {};
    this.regimeTransitions = [];
    this.eventTypeCounts = {};
    this.severityCounts = {};
    this.repeatedEventCounts = {};
    this.metricSamples = {};
  }

  record(row, organisms = [], extra = {}) {
    this.totalRowsRecorded += 1;
    this.#recordRow(row);
    this.#recordRegime(row);
    this.#recordMetricSamples(row);

    const autoEvents = [];
    if (row.regime !== this.prevRegime) autoEvents.push({ generation: row.generation, type: 'regime_change', regime: row.regime, severity: 'low', metrics: pick(row), description: `Regime changed ${this.prevRegime} -> ${row.regime}` });
    if (row.regime === 'VIOLET') autoEvents.push({ generation: row.generation, type: 'rare_geometry', regime: row.regime, severity: 'medium', metrics: pick(row), description: 'Rare structure emergence after instability.' });
    if (row.collapseRisk) autoEvents.push({ generation: row.generation, type: 'collapse_protection', regime: row.regime, severity: 'high', metrics: pick(row), description: 'RED collapse protection condition active.' });
    if (row.surprise > 0.25) autoEvents.push({ generation: row.generation, type: 'surprise_spike', regime: row.regime, severity: 'medium', metrics: pick(row), description: 'Surprise exceeded alert threshold.' });
    const biggest = organisms.reduce((m, o) => Math.max(m, o.mass), 0);
    if (biggest > this.largestMass + 25) {
      this.largestMass = biggest;
      autoEvents.push({ generation: row.generation, type: 'large_organism', regime: row.regime, severity: 'medium', metrics: pick(row), description: 'New large organism emerged.' });
    }
    this.#recordEvents([...autoEvents, ...(extra.events || [])]);
    if (extra.evolution) this.evolution = extra.evolution;
    this.prevRegime = row.regime;
  }

  #recordRow(row) {
    if (this.rows.length < this.maxRowsFull) {
      this.rows.push(row);
      return;
    }
    this.rows.shift();
    this.rows.push(row);
    if (this.totalRowsRecorded % this.sampleEveryAfterMax === 0) {
      this.sampledRows.push(row);
      if (this.sampledRows.length > this.maxSampledRows) this.sampledRows.shift();
    }
  }

  #recordRegime(row) {
    this.regimeCounts[row.regime] = (this.regimeCounts[row.regime] || 0) + 1;
    if (this.prevRegime !== row.regime) {
      this.regimeTransitions.push({ generation: row.generation, from: this.prevRegime, to: row.regime });
      if (this.regimeTransitions.length > this.maxRegimeTransitions) this.regimeTransitions.shift();
    }
  }

  #recordEvents(events) {
    for (const event of events) {
      this.events.push(event);
      this.recentEvents.push(event);
      if (this.recentEvents.length > 100) this.recentEvents.shift();
      const type = event.type || 'unknown';
      const severity = event.severity || 'low';
      this.eventTypeCounts[type] = (this.eventTypeCounts[type] || 0) + 1;
      this.severityCounts[severity] = (this.severityCounts[severity] || 0) + 1;
      const sig = `${type}|L${event.lineageId ?? 'na'}|G${event.genomeId ?? 'na'}`;
      this.repeatedEventCounts[sig] = (this.repeatedEventCounts[sig] || 0) + 1;
    }
    if (this.events.length > this.maxEvents) this.events.splice(0, this.events.length - this.maxEvents);
  }

  #recordMetricSamples(row) {
    for (const [k, v] of Object.entries(row)) {
      if (typeof v !== 'number' || Number.isNaN(v)) continue;
      if (!this.metricSamples[k]) this.metricSamples[k] = [];
      const arr = this.metricSamples[k];
      arr.push(v);
      if (arr.length > 50000) arr.shift();
    }
  }

  getSummary() {
    const metrics = ['surprise','entropy','compressionRatio','plasticity','aliveRatio','organismCount','largestOrganismMass','livingLineages','activeGenomes','oldestOrganismAge','localRegimeDiversity','predictionError','births','deaths','mutations','splitCount','mergeCount','deathCount'];
    const telemetryStats = Object.fromEntries(metrics.map((m) => [m, summarize(this.metricSamples[m] || [])]));
    const sortedPatterns = Object.entries(this.repeatedEventCounts).sort((a,b)=>b[1]-a[1]).slice(0,15).map(([signature,count])=>({signature,count}));
    return {
      totalRowsRecorded: this.totalRowsRecorded,
      retainedRows: this.rows.length,
      sampledRows: this.sampledRows.length,
      telemetryStats,
      regimeCounts: this.regimeCounts,
      regimeTransitions: {
        count: this.regimeTransitions.length,
        first: this.regimeTransitions.slice(0, 20),
        recent: this.regimeTransitions.slice(-20),
      },
      eventSummary: {
        eventTypeCounts: this.eventTypeCounts,
        severityCounts: this.severityCounts,
        recentEvents: this.recentEvents.slice(-100),
        topEventPatterns: sortedPatterns,
      },
    };
  }

  exportSummary(worldState = null, runMeta = {}, extras = {}) {
    const summary = this.getSummary();
    return {
      summary_export_version: '1.1',
      exported_at: new Date().toISOString(),
      run_meta: runMeta,
      manifest: {
        version: '0.4',
        experiment: 'eidos-life-evolutionary-world-layer',
        startedAt: this.startedAt,
      },
      settings: extras.settings || {},
      source_summary: {
        telemetry_rows_total: this.totalRowsRecorded,
        telemetry_rows_retained: this.rows.length,
        telemetry_rows_sampled: this.sampledRows.length,
      },
      final_world_compact: extras.finalWorldCompact || {},
      telemetry_stats: summary.telemetryStats,
      regime_counts: summary.regimeCounts,
      regime_transitions_compact: summary.regimeTransitions,
      event_summary: summary.eventSummary,
      top_events_recent: this.events.slice(-100),
      top_genomes: (this.evolution.genomes || []).slice(0, 30),
      top_lineages: (this.evolution.lineages || []).slice(0, 30),
      top_organisms: (this.evolution.organisms || []).slice(0, 30),
      telemetry_sample: this.sampledRows.slice(-500),
      world_state_reference: worldState ? { generation: worldState.generation, scenario: worldState.scenario } : null,
    };
  }

  exportBundle(worldState = null) { /* unchanged surface */
    const summary = {
      version: '0.4', generations: this.rows.length, peakSurprise: Math.max(0, ...this.rows.map(r => r.surprise)), violetFrames: this.rows.filter(r => r.regime === 'VIOLET').length, events: this.events.length,
      livingLineages: last(this.rows)?.livingLineages || 0, activeGenomes: last(this.rows)?.activeGenomes || 0, oldestOrganismAge: last(this.rows)?.oldestOrganismAge || 0, largestOrganismMass: last(this.rows)?.largestOrganismMass || 0,
    };
    const manifest = { version: '0.4', experiment: 'eidos-life-evolutionary-world-layer', startedAt: this.startedAt, exportedAt: new Date().toISOString() };
    return { manifest, summary, telemetry: this.rows, interestingEvents: this.events, evolution: this.evolution, worldState };
  }
}

function summarize(values) { if (!values.length) return { count:0,min:0,mean:0,p50:0,p90:0,p95:0,max:0 }; const s=[...values].sort((a,b)=>a-b); const mean=values.reduce((a,b)=>a+b,0)/values.length; const q=(p)=>s[Math.min(s.length-1,Math.floor((s.length-1)*p))]; return { count:values.length,min:s[0],mean,p50:q(0.5),p90:q(0.9),p95:q(0.95),max:s[s.length-1] }; }
function pick(r){return{surprise:r.surprise,entropy:r.entropy,compressionRatio:r.compressionRatio,novelty:r.novelty,predictionError:r.predictionError};}
function last(items){return items[items.length-1]||null;}
