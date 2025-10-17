Unreleased

- fix(train/iw): Coerce IW subsample inputs before SGD, reuse adversarial weights with temperature/clip controls, and add regression guard against string suburbs.
- fix(train/oof-cv): Stream CV metrics and remove OOF array materialization in model_training.py to avoid memory spikes on Windows; behavior and blend learning unchanged.
- fix(plot): Restore Actual vs Predicted plots by importing plotting functions before use and logging plot-time exceptions instead of silently skipping.
- fix(train/cv-metrics): Strip 'halflife_days' from CatBoost param dicts during CV metrics recompute and pre-final snapshot to prevent CatBoostRegressor.__init__ errors.
- feat(impute): Add BedConditionalMedianImputer to numeric pipeline (leak‑safe per‑Bed medians with global fallback); remove cleaning‑time numeric fills to ensure train↔serve parity.

- fix(model): Harden Ridge with AutoOrdinalOnObjects + SafeMedianImputer; unify Ridge DF-in builder across CV/final; stamp training_env in metadata; pin env to Python 3.12.5 with numpy==1.26.4, pandas==2.2.2, scikit-learn==1.7.0, scipy==1.14.1, catboost==1.2.8, shap==0.46.0, joblib==1.4.2, pyarrow==16.1.0, category-encoders==2.6.3, optuna==3.6.1.

- fix(model): Unify Ridge as DF-in with embedded preprocessor; fold-synchronous Ridge OOF; prune non-positive blend weights; truthful ridge_signature (expects_dataframe=True, n_in=len(feature_order)); clip cat_idx_final.
- perf(infer): Serve adapter reads ridge_signature and auto-selects DF vs matrix inputs for Ridge; supports legacy matrix-in artifacts; CatBoost-only short-circuit and serve-time gating unchanged.

- fix(train): Prevent accidental early exit by guarding FEATURES_ONLY/DRY_RUN_FEATURES behind ALLOW_FEATURES_ONLY and ignoring when OPTUNA_TRIALS>0.

- feat(serve): Implement standalone prediction parity with byte-for-byte copy, SHA-256 hash tracking, and observability logging (PREDICT_DEBUG=1); replaced delegation wrapper with full root predictor copy.
- ops(train): Post-training sync now computes and stores prediction.py SHA-256 hash in metadata.serve_bundle.code_sha and artifacts/Prediction/prediction.sha256 file for traceability.
- test(parity): Add test_standalone_code_parity.py to verify file hash equality and SHA-256 file presence; test_prediction_parity.py confirms identical predictions (≤1e-6) for same inputs.
- fix(serve): Add robust Ridge gating at serve-time (min-bed, abs/rel delta thresholds, global disable) with identical logic in standalone predictor.
- feat(model): Persist Ridge segment diagnostics and training-time gate; set CatBoost-only blend when Ridge underperforms segment criteria; metadata includes blend.ridge_allowed, blend.gate, and blend.segment_metrics.
- fix(model): Ensure halflife is excluded from CV params and persist final_halflife_days; remove redundant warnings in CV recompute.

- fix(serve): Keep standalone predictor in lockstep with pipeline; add sync script and parity regression test.
- fix(infer): Recalibrate serve-time prediction intervals with metadata targets and neighbor residuals; ensure final PI is printed once.
- fix(split): Enforce grouped walk-forward splits with group blocking across TRAIN/EARLY/TEST and within CV; persist per-fold WMAE CSV (folds_{hash}.csv).
- feat(feature-eng): Add RECENCY_V2 as-of anchors (rec2_* medians 30/90/180/365 + momentum) gated by RECENCY_V2=1; train=serve parity with closed='left'.
- feat(model): Seed bagging — train and persist ENSEMBLE_SIZE CatBoost seed pipelines; metadata.serve_bundle.catboost_seed_pipelines lists saved paths.
- perf(infer): prediction loader averages seed pipelines when present and falls back gracefully; backwards-compatible with base_pipelines and single pipeline.
- ops(repro): Snapshot pip freeze and run command per run; add schema_version and CV summary to metadata.

- feat(blend): Add Ridge probe (OOF under identical CV) and learn non-negative NNLS blend weights over CatBoost+Ridge; persist weights/pipelines and apply at serve with safe fallbacks.
- fix(clean): Unify target upper bound to 2,500 in domain filter to match earlier clamp, preventing loss of valid high-end rentals.
- feat(feature-eng): Make categorical cap tunable via FE_MAX_LEVELS (env) and persist the value to metadata for reproducibility.

- fix(schema): Canonicalize Furnishings/Pets/Garage values (and size aliases) so cleaned/model inputs carry normalized categories.
- fix(schema): Align land/floor size aliases so training and prediction share canonical columns; serve output now records both snake_case and spaced names.
- fix(infer): Preserve lat/lon through log helper so serve-time geo distances populate and geo features remain active at prediction time.
- fix(infer): Skip corrupt pipeline artifacts during prediction load, falling back to the most recent healthy joblib.

- fix(serve): Furnishings canonicalization lowercased to match training artifacts; avoids binning to OTHER and restores price sensitivity when toggling Furnished/Unfurnished/Partially Furnished.
- fix(cleaning): Furnishings canonicalizer now emits lowercase categories consistently in cleaned data.
- feat(model): Default monotone constraints include Floor Size (sqm) (when ENFORCE_MONOTONE=1), reducing inverted size effects seen on hold-out.

- fix(policy): Centralize log_* policy via `log_policy` module; training and serve reuse exports for new regression tests.
- fix(drift): Adversarial probe coerces mixed types and rejects empty TEST folds before weighting.
- fix(geo): Geo POI config now appends school categories and exposes bearings/accessibility toggles for parity.
- fix(split): _infer_gap_months exported and make_time_features ignores missing suburb/date columns.
- ops(test): Add pytest.ini to confine discovery to repo tests on Windows sandboxes.

- fix(split): Default hold-out now 2M TEST / 1M EARLY with month embargo; dynamic fallback adapts to short histories.
- feat(feature-eng): Enforce ALLOWED_LOG_FEATURES across train/serve; strip disallowed log_* columns and tests cover policy.
- feat(geo): Bearings/accessibility toggles default off; school distances consolidated into dist_school_min_km with metadata + serve parity.
- fix(drift): Adversarial probe/importance weighting now operate on scaled numeric matrices (saga LR), avoiding string cast failures.

- fix(split): Enforce month-label cutoff (train >=60% rows, hold >=25%), purge walk CV off hold months, and gate drift weights safely.

- feat(split): Dynamic month split, purged walk CV, adversarial weighting guardrails, and geo school parity for leak-safe hold-outs.

- feat(schema): Add robust alias + normalization for new rental_data.csv schema in training; clamps and casts per spec.
- fix(cleaning): data_cleaning now canonicalizes new schemas (aliases, dayfirst, clamps) for leak-safe downstream steps.
- fix(leak): Canonical alias renaming removes snake_case target columns so `last_rent_price` cannot enter feature set.
- fix(cv): Raise CatBoost subsample floor (>=0.7) to keep sampling enabled and avoid "Too few sampling units".
- fix(drift): `_adversarial_auc` coerces arrays to float so logistic regression no longer fails on string features.
- feat(logging): Track all removed properties per pipeline step (`plots/removed_properties.csv`).
- fix(drift): Increase adversarial logistic regression patience (`max_iter=5000`, `n_jobs=-1`) to avoid LBFGS early stop.
- feat(geo): Wire geo features/geocoding env config (train+serve parity, metadata, tests).
- perf(infer): Cache quantile pipelines in prediction keyed by metadata fingerprint.
- perf(geo): Cache POI CSV loads in geo_features for lower latency.
- fix(geo): main.py now wires GEO_POI_CSV and related CLI/env to training; auto-detects artifacts/poi_christchurch.csv so metadata.geo is populated and POIs are used.
- feat(geo): Add config.py defaults (GEO_POI_CSV, GEO_LAT_COL, GEO_LON_COL, GEO_RADII_KM, GEO_DECAY_KM, GEO_MAX_DECAY_KM, GEO_CATEGORIES); main.py reads these to enable POIs without CLI flags. Default radii now include 4.0km to cover high‑school zones.
 - fix(geo): Introduce GEO_FORCE_OFF hard switch in main.py/model_training.py/prediction.py to completely disable geo features even when config defaults are present.
 - fix(plots): SHAP summary now uses top‑K ranking, dynamic sizing, and saves both beeswarm and bar plots at high DPI for readability.
 - feat(features): PreInputAdapter now creates log features for Value/Size/DOM and zero‑indicators for size columns with non‑negative clipping to enforce train↔serve parity.
 - feat(features): Prefer log numerics by default (including Land/Improvement Value), with optional KEEP_RAW_WHEN_LOG override; pin key logs and zero flags to survive Top‑K selection.
 - fix(serve): prediction.py default PREFER_LOG_FOR updated to match training; log_policy default whitelist expanded to retain the new logs.
 - feat(predict): Add optional nearest comparables output (BallTree-based) via --neighbors, --neighbors-radius-km, and bed preference; visual only, no impact on predictions.
 - fix(blend): Stabilize Ridge blend by pinning key log features in CV folds and scaling inputs with MaxAbsScaler in Ridge pipelines (fold + final). Restores positive OOF R² and sensible blend weights.
 - feat(analysis): Add enhanced error analysis — dual overlay plot, residual-colored scatter, worst-error CSV, and segment error tables (by Bed, Property Type, Suburb, and floor-size bins); saved to plots/ with ds_hash.
 - feat(geo): Add CBD to required POI categories so `dist_cbd_km` is produced whenever a category whitelist is supplied.
- fix(cli): main.py resolves date/target/suburb using alias sets (new + legacy).
- feat(features): Always retain raw 'Bath' alongside 'bed_bath_ratio' and include 'Garage Parks' (numeric) when present; exclude logs for Bath/Car/Garage from model inputs.
- perf(train): Optuna search space narrowed and coupled (depth 7–9, iterations 2400–4500, leaf size tied to depth); halflife choices refined to [0,45,90,120,150,210].
- perf(train): Switch to TPESampler(multivariate/group) + MedianPruner(n_warmup_steps=3) for faster, more stable HPO.
- feat(train): Adopt production baseline CatBoost params for final fit (and quantile models), including od_wait=200 and consistent subsampling.
- feat(train): Add FINAL_USE_TUNED toggle to base final fit on Optuna best_params; iterations capped by MAX_FINAL_ITER (set to 5000 to extend).
- fix(train): Force headless Matplotlib backend ('Agg') to prevent Tkinter/Tcl crashes during HPO and training on Windows.
- feat(geo): add resumable geocode runner, share query normalization, and merge cached coordinates in train/serve pipelines.
- fix(train): tolerate small CV splits by clamping indices and skipping destructive filters when they empty data.
- feat(geo): Structured-first NZ geocoding with suburb/city/state; lean fallbacks; deterministic cache key; respects viewbox+bounds to cut requests and improve hit-rate.
 - fix(geo): geocode_properties_planA now pre-merges artifacts/geocode_query_results.csv and appends newly resolved rows for resume parity with unique-queries flow.
 - perf(geo): geocode_properties_planA processes unresolved rows in configurable chunks (GEOCODE_CHUNK_SIZE, default 300) to reduce Python overhead while respecting Nominatim rate limits.
 - fix(geo): PlanA appends results to geocode_query_results.csv with a stable 5-column schema [query, used_query, latitude, longitude, provider] to prevent column-shift and row drops when merging.

- fix(train): Skip skew/log transforms for near-constant features to avoid precision loss warnings during preprocessing.
- fix(clean): Harden dateutil fallback to skip placeholder NaT strings and only drop rows that remain unparseable.
- fix(eda): Remove Plotly text_auto dependency and inject manual annotations for heatmaps.
- fix(train): Teach LogTargetWrapper/AverageEnsemble to advertise fitted state for sklearn pipelines to prevent future warnings.
- fix(geo): limit school features, add university distance, clip via env; drop raw lat/lon by policy.
- fix(features): shared policy drops unwanted month/car flags, applies env-driven numeric exclusions and distance clipping.
- feat(model): monotone constraints configurable via MONO_INC/MONO_DEC for final CatBoost fit.
- feat(geo): emit per-category distance features (CBD, airport, hospital, etc.) without count/access clutter; tests updated.
- fix(predict): VWAP calculation now falls back to all historical data by bed when specific bed history is missing, avoiding serve-time errors.
- fix(predict): keep lat/lon through geocoding, auto-geocode missing coordinates from address input, and support new CLI options (city/category/lat/long).
- fix(predict): align CLI/base schema with training (agency/category/city/etc.), accept furnishings/parking inputs, and auto-geocode missing coordinates using address.
- fix(predict): auto-geocode missing lat/lon, append results to geocode cache, and expose CLI for furnishings/geo fields.
- fix(geo): default NOMINATIM_EMAIL to sylabis@gmail.com so prediction geocoding works out of the box.
- fix(predict): geocoder now logs failures, writes successful lookups back to geocode_query_results.csv, and uses per-row __geo_query__ to ensure deterministic distance features.
- fix(predict): fallback to default POI CSV when metadata missing, and log successful geocodes with cache updates.


2025-10-12 fix(serve): Remove standalone predictor usage; enforce train→serve log policy parity
- remove(serve): Deprecate and stop training-time sync of artifacts/Prediction; root predictor is now the single source of truth.
- test: Drop standalone code parity test; convert parity test to root-only sanity check using historical data.
- fix(serve): Enforce keep_raw_when_log from training metadata at serve-time to prevent feature drop/magnitude drift; add schema mismatch warning when feature_order partially matches.

2025-10-12 fix(blend/serve): Stabilize Ridge in blend and avoid NaN failures
- fix(train): Add SimpleImputer(strategy='median') to Ridge CV and final pipelines (after categorical encoding) so Ridge handles any remaining NaNs.
- fix(serve): For Ridge predictions in blending, apply the main preprocessor transform before calling the Ridge pipeline (which expects preprocessed numeric input).

2025-10-14 perf(train): Harden CV objective, prevent NaN trials; prune invalid folds; drop all-NaN numerics safely

- Fix UnboundLocalError by ensuring preprocessor is always constructed.
- Remove duplicated all-NaN drop code; apply once per fold and on full TRAIN.
- Add guards in Optuna CV objective: skip empty/invalid folds, catch train/predict errors, and prune trials with no valid folds.
- Add fallback defaults when Optuna has no completed trials.
- Keep numeric imputation stable by excluding all-NaN columns from numeric block.
