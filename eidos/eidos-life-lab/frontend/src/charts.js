const SERIES = [
  { key: 'alive', color: '#8ee6a7', label: 'Alive' },
  { key: 'largestComponent', color: '#f0d78b', label: 'Largest' },
  { key: 'birthCandidates', color: '#74c7e6', label: 'Births' },
  { key: 'densityScaled', color: '#e68f74', label: 'Density x1000' }
];

export class TrendChart {
  constructor(canvas, maxPoints = 160) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.maxPoints = maxPoints;
    this.points = [];
  }

  add(metrics) {
    this.points.push({
      alive: metrics.alive,
      largestComponent: metrics.largestComponent,
      birthCandidates: metrics.birthCandidates,
      densityScaled: metrics.density * 1000,
      regime: metrics.sentinelRegime
    });
    if (this.points.length > this.maxPoints) {
      this.points.shift();
    }
    this.draw();
  }

  draw() {
    const ctx = this.context;
    const width = this.canvas.width;
    const height = this.canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#091010';
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(255,255,255,0.09)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i += 1) {
      const y = (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    if (this.points.length < 2) {
      this.drawLegend();
      return;
    }
    const maxValue = Math.max(1, ...this.points.flatMap((point) => SERIES.map((series) => point[series.key])));
    for (const series of SERIES) {
      ctx.strokeStyle = series.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      this.points.forEach((point, index) => {
        const x = (index / (this.points.length - 1)) * (width - 16) + 8;
        const y = height - 20 - (point[series.key] / maxValue) * (height - 34);
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }
    this.drawLegend();
  }

  drawLegend() {
    const ctx = this.context;
    ctx.font = '11px system-ui, sans-serif';
    SERIES.forEach((series, index) => {
      const x = 10 + index * 78;
      ctx.fillStyle = series.color;
      ctx.fillRect(x, 10, 10, 2);
      ctx.fillStyle = 'rgba(230,238,232,0.76)';
      ctx.fillText(series.label, x + 14, 14);
    });
  }
}
