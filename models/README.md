# Trained model artifacts

These are the trained classifier artifacts used by the **heartbeat (cardiac
viability)** pipeline.

| File | Description | Used by |
|------|-------------|---------|
| `alive_weak_strong_xgb.json` | 3-class XGBoost model (Dead / Weak / Strong). The production inference script collapses Weak+Strong into **Alive** for the Alive/Dead call. **Default model** loaded by the pipeline. | `sam_pipeline/heartbeat_pipeline_xgboost_CLEAN.py` |
| `alive_dead_binary_xgb.json` | Dedicated binary (Alive / Dead) XGBoost model, reported alongside the collapsed 3-class model in the paper. | Optional; pass via `--xgb_model_path` with `--fallback_model_mode binary`. |
| `alive_weak_strong_xgb_features.txt` | Ordered list of the predictor names expected by the model (the fixed-length fish feature vector). | Reference / validation. |

## Default path

`heartbeat_pipeline_xgboost_CLEAN.py` loads `models/alive_weak_strong_xgb.json`
from this checkout by default. Override it with `--xgb_model_path`.

> Legacy `.joblib` blob-level models that were present in the original working
> directory are **not** part of the production pipeline and are intentionally
> omitted from this release.
