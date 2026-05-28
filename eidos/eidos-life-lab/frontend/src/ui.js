export function formatNumber(value, digits = 3) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'NA';
  }
  return value.toFixed(digits);
}

export function updateMetricsGrid(container, metrics) {
  const rows = [
    ['Components', metrics.components],
    ['Largest component', metrics.largestComponent],
    ['Birth candidates', metrics.birthCandidates],
    ['Active genomes', metrics.activeGenomes],
    ['Active lineages', metrics.activeLineages],
    ['Energy mean', formatNumber(metrics.aliveEnergyMean, 3)],
    ['Memory mean', formatNumber(metrics.aliveMemoryMean, 3)],
    ['Signal mean', formatNumber(metrics.aliveSignalMean, 3)],
    ['Nutrient mean', formatNumber(metrics.aliveNutrientMean, 3)],
    ['Waste mean', formatNumber(metrics.aliveWasteMean, 3)],
    ['Stress mean', formatNumber(metrics.aliveStressMean, 3)]
  ];
  container.innerHTML = rows
    .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
    .join('');
}

export function renderEvents(container, events) {
  const recent = [...events].slice(-12).reverse();
  container.innerHTML = recent.length
    ? recent
        .map(
          (event) => `
            <article>
              <span>gen ${event.generation}</span>
              <strong>${event.kind}</strong>
              <p>${event.message}</p>
            </article>
          `
        )
        .join('')
    : '<p class="empty-state">No events yet.</p>';
}

export function regimeClass(regime) {
  return `sentinel-badge regime-${String(regime || '').toLowerCase().replaceAll('_', '-')}`;
}
