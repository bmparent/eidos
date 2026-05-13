const clamp01 = value => Math.max(0, Math.min(1, value));

export class EcologyFields {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this.size = width * height;
    this.scratch = new Float32Array(this.size);
  }

  reset(engine) {
    engine.nutrientField.fill(0.72);
    engine.wasteField.fill(0);
    engine.signalField.fill(0);
    engine.anomalyField.fill(0);
    engine.memoryField.fill(0);
  }

  contextAt(engine, i, genome, localModifier = {}) {
    const nutrient = engine.nutrientField[i];
    const waste = engine.wasteField[i];
    const anomaly = engine.anomalyField[i];
    const memory = engine.memoryField[i];
    const stress = engine.stress[i];
    const mutationPressure = clamp01(
      anomaly * 0.45 +
      waste * 0.25 +
      stress * 0.25 +
      (localModifier.mutationBoost || 0)
    );
    const stability = clamp01(
      memory * (genome?.traits.memoryAffinity ?? 0.5) +
      nutrient * 0.28 -
      waste * 0.22 -
      anomaly * 0.2
    );
    return { nutrient, waste, anomaly, memory, stress, mutationPressure, stability };
  }

  update(engine, genomeRegistry, { intervention = 'passive', collapseRisk = 0 } = {}) {
    const { alive, age, energy, genomeId, nutrientField, wasteField, signalField, anomalyField, memoryField, stress, size } = engine;
    for (let i = 0; i < size; i++) {
      const genome = genomeRegistry.get(genomeId[i]);
      const traits = genome?.traits;
      if (alive[i]) {
        const uptake = 0.006 + (traits?.energyUptake ?? 0.5) * 0.018;
        nutrientField[i] = Math.max(0, nutrientField[i] - uptake);
        wasteField[i] = clamp01(wasteField[i] + 0.004 + age[i] / 65535 + energy[i] * 0.006);
        signalField[i] = clamp01(signalField[i] + (traits?.signalEmission ?? 0.45) * 0.035);
        memoryField[i] = clamp01(memoryField[i] + 0.015 + (traits?.memoryAffinity ?? 0.45) * 0.035);
      } else {
        const recovery = collapseRisk && intervention !== 'passive' ? 0.018 : 0.007;
        nutrientField[i] = clamp01(nutrientField[i] + recovery);
        wasteField[i] *= 0.982;
        signalField[i] *= 0.94;
        memoryField[i] *= 0.982;
      }
      anomalyField[i] *= intervention === 'guardian' ? 0.91 : 0.94;
      stress[i] = clamp01(stress[i] + wasteField[i] * 0.012 + anomalyField[i] * 0.02);
    }

    this.diffuse(nutrientField, 0.08, 0.997);
    this.diffuse(wasteField, 0.055, 0.992);
    this.diffuse(signalField, 0.12, 0.965);
    this.diffuse(anomalyField, 0.045, 0.955);
    this.diffuse(memoryField, 0.035, 0.988);
  }

  diffuse(field, rate, decay) {
    const { width, height, scratch } = this;
    for (let y = 0; y < height; y++) {
      const ym = ((y - 1 + height) % height) * width;
      const y0 = y * width;
      const yp = ((y + 1) % height) * width;
      for (let x = 0; x < width; x++) {
        const xm = (x - 1 + width) % width;
        const xp = (x + 1) % width;
        const i = y0 + x;
        const neighborAverage = (field[y0 + xm] + field[y0 + xp] + field[ym + x] + field[yp + x]) * 0.25;
        scratch[i] = clamp01((field[i] + (neighborAverage - field[i]) * rate) * decay);
      }
    }
    field.set(scratch);
  }
}
