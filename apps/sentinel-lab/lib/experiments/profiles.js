// Kept in parity with sentinel_runner.profiles by the cross-language contract test.
export const ENGINE_PROFILES = {
  cpu_engineering: {
    label: "Standard engine", reservoir: 256, hippocampus_dim: 2048, fractal_bands: 1, trace_seal_enabled: false,
    description: "The established CPU profile. Reservoir prediction, episodic memory, adaptive learning and thermodynamic regulation are active.",
  },
  cpu_mechanisms: {
    label: "Multi-scale + TraceSeal study", reservoir: 256, hippocampus_dim: 2048, fractal_bands: 4, trace_seal_enabled: true,
    description: "Adds four memory time scales and residual-subspace filtering. Experimental: compare with the standard profile using the same data and seed; benefit is unproven.",
  },
  full_capacity: {
    label: "Full-size reservoir", reservoir: 2000, hippocampus_dim: 10000, fractal_bands: 1, trace_seal_enabled: false,
    description: "Uses the original 2,000-unit reservoir and 10,000-dimensional memory. Requires a dedicated runner explicitly enabled for this resource budget.",
  },
};

export function engineProfile(spec) {
  const name = spec.engine.executionProfile ?? "cpu_engineering";
  if (!Object.hasOwn(ENGINE_PROFILES, name)) throw new Error("Unsupported engine execution profile.");
  return ENGINE_PROFILES[name];
}

export function reviseDatasetSource(current, patch) {
  const changed = ["ref", "version", "file"].some((key) => key in patch && patch[key] !== current.dataset[key]);
  return { ...current, dataset: { ...current.dataset, ...(changed ? { expectedSha256: undefined } : {}), ...patch } };
}

export function selectDataset(current, result) {
  return { ...current, dataset: { provider: "kaggle", ref: result.ref, version: result.version ?? 0, file: "" } };
}
