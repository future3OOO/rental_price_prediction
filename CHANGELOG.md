Unreleased

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

