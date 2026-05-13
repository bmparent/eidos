const REGIME_NAMES = ['CALIBRATING', 'GREEN', 'AMBER', 'RED', 'BLUE', 'VIOLET'];
const REGIME_CODE = Object.fromEntries(REGIME_NAMES.map((name, index) => [name, index]));

const MODIFIERS = {
  CALIBRATING: { birthBoost: 0, surviveBoost: 0, mutationBoost: 0.01, nutrientBoost: 0 },
  GREEN: { birthBoost: 0.01, surviveBoost: 0.02, mutationBoost: 0, nutrientBoost: 0 },
  AMBER: { birthBoost: 0.02, surviveBoost: -0.02, mutationBoost: 0.08, nutrientBoost: -0.01 },
  RED: { birthBoost: 0.04, surviveBoost: 0.03, mutationBoost: 0.025, nutrientBoost: 0.035 },
  BLUE: { birthBoost: 0.005, surviveBoost: 0.04, mutationBoost: 0.015, nutrientBoost: 0.005 },
  VIOLET: { birthBoost: 0.05, surviveBoost: -0.01, mutationBoost: 0.16, nutrientBoost: 0.01 },
};

const clamp01 = value => Math.max(0, Math.min(1, value));

export class LocalRegimeMap {
  constructor(width, height, { tilesX = 12, tilesY = 12 } = {}) {
    this.width = width;
    this.height = height;
    this.tilesX = tilesX;
    this.tilesY = tilesY;
    this.map = new Uint8Array(tilesX * tilesY).fill(REGIME_CODE.CALIBRATING);
    this.previousAlive = new Uint8Array(width * height);
    this.tileWidth = Math.ceil(width / tilesX);
    this.tileHeight = Math.ceil(height / tilesY);
    this.lastVioletCount = 0;
  }

  reset() {
    this.map.fill(REGIME_CODE.CALIBRATING);
    this.previousAlive.fill(0);
    this.lastVioletCount = 0;
  }

  update(snapshot, { novelty = 0, generation = 0 } = {}) {
    const { alive, anomalyField, memoryField, stress, width, height } = snapshot;
    let violet = 0;
    for (let ty = 0; ty < this.tilesY; ty++) {
      for (let tx = 0; tx < this.tilesX; tx++) {
        let live = 0, flips = 0, anomaly = 0, memory = 0, stressSum = 0, cells = 0;
        const x0 = tx * this.tileWidth;
        const y0 = ty * this.tileHeight;
        const x1 = Math.min(width, x0 + this.tileWidth);
        const y1 = Math.min(height, y0 + this.tileHeight);
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const i = y * width + x;
            live += alive[i];
            flips += alive[i] === this.previousAlive[i] ? 0 : 1;
            anomaly += anomalyField?.[i] || 0;
            memory += memoryField?.[i] || 0;
            stressSum += stress?.[i] || 0;
            cells++;
          }
        }
        const density = live / Math.max(1, cells);
        const change = flips / Math.max(1, cells);
        const anomalyPressure = anomaly / Math.max(1, cells);
        const memoryResidue = memory / Math.max(1, cells);
        const localStress = stressSum / Math.max(1, cells);
        const entropy = density === 0 || density === 1 ? 0 : -(density * Math.log2(density) + (1 - density) * Math.log2(1 - density));
        let regime = 'GREEN';
        if (generation < 18) regime = 'CALIBRATING';
        else if (density < 0.018 || density > 0.7 || localStress > 0.62) regime = 'RED';
        else if (change > 0.18 && anomalyPressure + novelty > 0.22) regime = 'VIOLET';
        else if (change > 0.12 || anomalyPressure > 0.22) regime = 'AMBER';
        else if (entropy > 0.72 && memoryResidue > 0.16) regime = 'BLUE';
        if (regime === 'VIOLET') violet++;
        this.map[ty * this.tilesX + tx] = REGIME_CODE[regime];
      }
    }
    this.previousAlive.set(alive);
    const violetEmergence = violet > this.lastVioletCount;
    this.lastVioletCount = violet;
    return { violetCount: violet, violetEmergence };
  }

  regimeAtCell(x, y) {
    const tx = Math.min(this.tilesX - 1, Math.floor(x / this.tileWidth));
    const ty = Math.min(this.tilesY - 1, Math.floor(y / this.tileHeight));
    return REGIME_NAMES[this.map[ty * this.tilesX + tx]];
  }

  modifierAtIndex(index, width = this.width) {
    const x = index % width;
    const y = Math.floor(index / width);
    return MODIFIERS[this.regimeAtCell(x, y)];
  }

  diversity() {
    return new Set(this.map).size;
  }

  export() {
    return {
      width: this.tilesX,
      height: this.tilesY,
      regimes: Array.from(this.map, code => REGIME_NAMES[code]),
    };
  }
}

export { REGIME_NAMES, REGIME_CODE, MODIFIERS, clamp01 };
