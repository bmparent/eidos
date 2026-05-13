export class PredictionGhost {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this.size = width * height;
    this.predicted = new Uint8Array(this.size);
    this.errorField = new Float32Array(this.size);
    this.sparks = [];
  }

  predict(engine, rule, localRegimes = null) {
    const { alive, genomeId, genomeRegistry, width, height } = engine;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = engine.idx(x, y);
        const neighbors = engine.countNeighbors(x, y);
        const genome = genomeRegistry.get(genomeId[i]);
        const modifier = localRegimes?.modifierAtIndex?.(i, width) || {};
        const birthBias = genome?.traits.birthBias ?? 0.5;
        const surviveBias = genome?.traits.surviveBias ?? 0.5;
        const predictsBirth = rule.birth.includes(neighbors) || (neighbors === 2 && birthBias + (modifier.birthBoost || 0) > 0.58);
        const predictsSurvival = rule.survive.includes(neighbors) || (neighbors === 4 && surviveBias + (modifier.surviveBoost || 0) > 0.66);
        this.predicted[i] = alive[i] ? (predictsSurvival ? 1 : 0) : (predictsBirth ? 1 : 0);
      }
    }
    return this.predicted;
  }

  compare(actualAlive, generation = 0) {
    let misses = 0;
    this.sparks = [];
    for (let i = 0; i < this.size; i++) {
      const miss = this.predicted[i] === actualAlive[i] ? 0 : 1;
      this.errorField[i] = this.errorField[i] * 0.82 + miss;
      if (miss) {
        misses++;
        if (this.sparks.length < 80) this.sparks.push({ index: i, x: i % this.width, y: Math.floor(i / this.width), generation });
      } else {
        this.errorField[i] *= 0.95;
      }
    }
    return {
      predicted: this.predicted,
      errorField: this.errorField,
      sparks: this.sparks,
      predictionError: misses / this.size,
    };
  }
}
