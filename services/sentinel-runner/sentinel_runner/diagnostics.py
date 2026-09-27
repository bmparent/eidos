"""Bounded engine observations without input values, labels, or text snippets."""
import math

TELEMETRY_FIELDS = (
    "step", "is_surprise", "best_err", "z", "z_thresh_eff", "ratio", "plasticity",
    "dominance", "state_entropy", "spectral_entropy", "hipp_sim", "hipp_chi",
    "hipp_write", "lr_scale_raw", "lr_scale_eff", "thermo_energy", "thermo_rho",
    "thermo_temp", "thermo_lambda", "ts_err", "ts_expl", "ts_cond",
)


def engine_telemetry(step_rows):
    for row in step_rows:
        yield {key: value if isinstance(value, (bool, int, float)) and math.isfinite(value) else None
               for key in TELEMETRY_FIELDS if (value := row.get(key)) is not None}


def summarize_engine(step_rows, engine_receipt, geometry=None):
    config = engine_receipt["effective_config"]
    statistics = {}
    for key in TELEMETRY_FIELDS:
        values = [float(row[key]) for row in step_rows
                  if type(row.get(key)) in (int, float) and math.isfinite(row[key])]
        if values:
            statistics[key] = {"samples": len(values), "mean": sum(values) / len(values), "min": min(values), "max": max(values)}
    return {
        "schema": "eidos.sentinel-runner.diagnostics.v1",
        "execution_profile": engine_receipt["execution_profile"],
        "code_sha256": engine_receipt["code_sha256"],
        "reservoir_units": config["reservoir"],
        "memory_dimensions": config["hippocampus_dim"],
        "leak_bands": config["fractal_bands"],
        "trace_seal_enabled": config["trace_seal_enabled"],
        "thermodynamics_enabled": config["thermo_enabled"],
        "processed_rows": len(step_rows),
        "surprise_rows": sum(row.get("is_surprise") is True for row in step_rows),
        "memory_writes": sum(row.get("hipp_write") is True for row in step_rows),
        "statistics": statistics,
        "trace": list(engine_telemetry(step_rows[::max(1, math.ceil(len(step_rows) / 240))])),
        "trace_sampling": "Uniform stride, at most 240 recorded rows; full telemetry in engine_trace.jsonl. Missing observations are omitted, never interpolated.",
        "geometry": geometry,
        "scope": "Post-warmup calibration and evaluation telemetry; no held-out rows. Activity does not establish mechanism benefit.",
    }
