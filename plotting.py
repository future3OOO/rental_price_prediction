# plotting.py

import logging
import os
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

###############################################################################
# (A) Standard (Non-Interactive) Plotting Functions
###############################################################################

def plot_feature_importance(
    model,
    feature_names,
    plot_dir,
    importance_type='PredictionValuesChange',
    top_n=20
):
    """
    Plot and save feature importance based on the specified importance_type.
    """
    try:
        importance = model.get_feature_importance(type=importance_type)
        if isinstance(importance, np.ndarray):
            importance = importance.tolist()

        model_feature_names = model.feature_names_

        logging.info(f"Length of model_feature_names: {len(model_feature_names)}")
        logging.info(f"Length of importance: {len(importance)}")

        # Synchronize lengths if needed
        if len(model_feature_names) != len(importance):
            logging.warning("Mismatch in model_feature_names vs. importance lengths. Truncating.")
            min_len = min(len(model_feature_names), len(importance))
            model_feature_names = model_feature_names[:min_len]
            importance = importance[:min_len]

        df_imp = pd.DataFrame({
            'Feature': model_feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        if df_imp.empty:
            logging.error("Feature importance DataFrame is empty.")
            return

        df_imp = df_imp.head(top_n)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title(f"Top {top_n} Feature Importance ({importance_type})", fontsize=16)
        plt.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        fname = os.path.join(plot_dir, f'feature_importance_{importance_type}.png')
        plt.savefig(fname)
        plt.close()
        logging.info(f"Feature importance plot saved => {fname}")

    except Exception as e:
        logging.error(f"Error in plot_feature_importance: {str(e)}")
        logging.error(traceback.format_exc())


def plot_learning_curve(
    estimator,
    X,
    y,
    plot_dir,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring='neg_mean_absolute_error',
    fit_params=None
):
    """
    Plot and save the learning curve of the estimator.
    """
    try:
        train_sizes_out, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            scoring=scoring,
            fit_params=fit_params
        )

        # Convert negative MAE to positive for interpretability
        train_mean = -np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = -np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(12, 8))
        plt.title("Learning Curve", fontsize=16)
        plt.xlabel("Training Examples", fontsize=13)
        plt.ylabel("Mean Absolute Error", fontsize=13)
        plt.grid(True)

        # Plot training
        plt.fill_between(train_sizes_out, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.plot(train_sizes_out, train_mean, 'o-', color="r", label="Training score")

        # Plot validation
        plt.fill_between(train_sizes_out, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        plt.plot(train_sizes_out, test_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best", fontsize=12)
        plt.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        fname = os.path.join(plot_dir, 'learning_curve.png')
        plt.savefig(fname)
        plt.close()
        logging.info(f"Learning curve plot saved => {fname}")

    except Exception as e:
        logging.error(f"Error in plot_learning_curve: {str(e)}")
        logging.error(traceback.format_exc())


def plot_residuals(y_true, y_pred, plot_dir):
    """
    Plot residual analysis: scatter (predicted vs residuals) + distribution.
    """
    try:
        residuals = y_true - y_pred
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=ax1)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.set_title('Residual Plot', fontsize=14)
        ax1.grid(True)

        # Distribution
        sns.histplot(residuals, kde=True, bins=40, ax=ax2)
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_title('Residual Distribution', fontsize=14)
        ax2.grid(True)

        plt.suptitle(f"Residual Analysis\nMAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}", fontsize=14)
        plt.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        fpath = os.path.join(plot_dir, 'residual_analysis.png')
        fig.savefig(fpath)
        plt.close(fig)
        logging.info(f"Residual analysis plot saved => {fpath}")

    except Exception as e:
        logging.error(f"Error in plot_residuals: {str(e)}")
        logging.error(traceback.format_exc())


def plot_actual_vs_predicted(y_true, y_pred, plot_dir, filename: str | None = None):
    """
    Scatter of Actual vs Predicted with a reference line + metrics.
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, ax=ax)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title('Actual vs Predicted', fontsize=14)
        ax.grid(True)

        textstr = f"MAE={mae:.2f}\nRMSE={rmse:.2f}\nR²={r2:.2f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(
            0.05, 0.95, textstr,
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props
        )

        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        out_name = filename if filename else 'actual_vs_predicted.png'
        fpath = os.path.join(plot_dir, out_name)
        plt.savefig(fpath)
        # Also write canonical stable name for dashboards/tools
        try:
            canonical = os.path.join(plot_dir, 'actual_vs_predicted.png')
            if os.path.abspath(fpath) != os.path.abspath(canonical):
                import shutil as _sh
                _sh.copyfile(fpath, canonical)
        except Exception:
            pass
        plt.close(fig)
        logging.info(f"Actual vs Predicted plot saved => {fpath}")
    except Exception as e:
        logging.error(f"Error in plot_actual_vs_predicted: {str(e)}")
        logging.error(traceback.format_exc())


def plot_actual_vs_predicted_dual(y_true, y_pred, plot_dir: str, filename: str | None = None):
    """Overlay Real (blue) vs Predict (red) points for the same index.

    - Adds y=x reference line
    - Uses light alpha to reveal density
    """
    try:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        n = min(len(y_true), len(y_pred))
        plt.figure(figsize=(8, 5))
        plt.scatter(range(n), y_true[:n], color='tab:blue', alpha=0.6, s=18, label='Real')
        plt.scatter(range(n), y_pred[:n], color='tab:red', alpha=0.6, s=18, label='Predict')
        plt.title('Real vs Predict (overlay)')
        plt.xlabel('Row index (TEST order)')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        fname = filename if filename else 'actual_vs_predicted_dual.png'
        outp = os.path.join(plot_dir, fname)
        plt.savefig(outp, dpi=150)
        # Also write canonical stable name
        try:
            canonical = os.path.join(plot_dir, 'actual_vs_predicted_dual.png')
            if os.path.abspath(outp) != os.path.abspath(canonical):
                import shutil as _sh
                _sh.copyfile(outp, canonical)
        except Exception:
            pass
        plt.close()
    except Exception as e:
        logging.warning(f"dual Real/Predict plot failed: {e}")


def plot_residual_scatter(y_true, y_pred, df: pd.DataFrame | None, plot_dir: str, filename: str):
    """Scatter of Actual vs Pred with points colored by abs% error and sized by error.

    If df provided and contains 'Bed', annotate colorbar title accordingly.
    """
    try:
        yt = np.asarray(y_true, float).reshape(-1)
        yp = np.asarray(y_pred, float).reshape(-1)
        n = min(len(yt), len(yp))
        yt, yp = yt[:n], yp[:n]
        ape = np.zeros(n)
        with np.errstate(divide='ignore', invalid='ignore'):
            mask = yt > 0
            ape[mask] = np.abs(yp[mask] - yt[mask]) / yt[mask]
        sz = (np.clip(ape, 0, 1.5) * 60 + 8)
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(yt, yp, c=ape, s=sz, cmap='viridis', alpha=0.65, edgecolors='none')
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        plt.plot(lim, lim, 'k--', linewidth=1, alpha=0.6)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted (colored by |APE|)')
        cbar = plt.colorbar(sc)
        cbar.set_label('|APE|')
        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        outp = os.path.join(plot_dir, filename)
        plt.savefig(outp, dpi=150)
        # Also write canonical stable name
        try:
            canonical = os.path.join(plot_dir, 'residual_scatter.png')
            if os.path.abspath(outp) != os.path.abspath(canonical):
                import shutil as _sh
                _sh.copyfile(outp, canonical)
        except Exception:
            pass
        plt.close()
    except Exception as e:
        logging.warning(f"residual scatter failed: {e}")


def save_error_tables(
    df_test: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    plot_dir: str,
    stem: str,
) -> dict:
    """Create error CSVs to identify hard segments and cases.

    Returns dict of generated paths.
    """
    out: dict = {}
    try:
        y = np.asarray(y_true, float).reshape(-1)
        p = np.asarray(y_pred, float).reshape(-1)
        n = min(len(y), len(p), len(df_test))
        df = df_test.iloc[:n].copy()
        df['y_true'] = y[:n]
        df['y_pred'] = p[:n]
        df['abs_err'] = np.abs(df['y_true'] - df['y_pred'])
        with np.errstate(divide='ignore', invalid='ignore'):
            df['ape'] = np.where(df['y_true'] > 0, df['abs_err'] / df['y_true'], np.nan)
        # Worst 100 by absolute error
        worst = df.sort_values('abs_err', ascending=False).head(100)
        os.makedirs(plot_dir, exist_ok=True)
        worst_path = os.path.join(plot_dir, f'worst_errors_{stem}.csv')
        worst.to_csv(worst_path, index=False)
        out['worst_errors'] = worst_path

        # Segment metrics: by Bed, Property Type, Suburb (top 30), floor/floor bins
        def _agg(group):
            return pd.Series({
                'n': int(group.shape[0]),
                'MAE': float(group['abs_err'].mean()),
                'MedAE': float(group['abs_err'].median()),
                'MAPE': float(group['ape'].mean(skipna=True) * 100.0),
            })
        segs = []
        if 'Bed' in df.columns:
            bed = df.groupby(df['Bed'].astype(str)).apply(_agg).reset_index().rename(columns={'Bed':'segment'})
            bed['group']='Bed'
            segs.append(bed)
        if 'Property Type' in df.columns:
            pt = df.groupby(df['Property Type'].astype(str)).apply(_agg).reset_index().rename(columns={'Property Type':'segment'})
            pt['group']='Property Type'
            segs.append(pt)
        if 'Suburb' in df.columns:
            sb = df.groupby(df['Suburb'].astype(str)).apply(_agg).reset_index().rename(columns={'Suburb':'segment'})
            sb = sb.sort_values('n', ascending=False).head(30)
            sb['group']='Suburb'
            segs.append(sb)
        if 'Floor Size (sqm)' in df.columns:
            fbin = pd.to_numeric(df['Floor Size (sqm)'], errors='coerce')
            bins = pd.qcut(fbin, q=5, duplicates='drop')
            fb = df.groupby(bins).apply(_agg).reset_index().rename(columns={'Floor Size (sqm)':'segment'})
            fb['segment'] = fb['index'].astype(str); fb = fb.drop(columns=['index'])
            fb['group']='Floor Size (bins)'
            segs.append(fb)
        if segs:
            seg_all = pd.concat(segs, axis=0, ignore_index=True)
            seg_path = os.path.join(plot_dir, f'error_segments_{stem}.csv')
            seg_all.to_csv(seg_path, index=False)
            out['segments'] = seg_path
    except Exception as e:
        logging.warning(f"error tables failed: {e}")
    return out
def save_shap_summary_catboost(
    model,
    X,
    feature_names,
    plot_dir: str,
    stem: str | None = None,
    artifact_path: str | None = None,
) -> tuple[str | None, str | None]:
    """Save high‑quality SHAP summaries (beeswarm + bar) with sane sizing.

    Best‑practice defaults:
    - Use TreeExplainer on CatBoost model
    - Limit to top_k features by mean(|SHAP|) to keep plots readable
    - Beeswarm colored by feature values, plus compact bar summary
    - Dynamic figure height; DPI=200 for clarity
    Returns (beeswarm_path, bar_path)
    """
    shap_art, shap_plot = None, None
    try:
        import shap
        import numpy as _np
        import matplotlib.pyplot as plt

        # Ensure directory exists
        os.makedirs(plot_dir, exist_ok=True)

        # Convert X to DataFrame with names for coloring in beeswarm
        names = list(feature_names) if feature_names is not None else None
        if hasattr(X, "shape") and names is not None and len(names) == getattr(X, "shape", [0])[1]:
            try:
                X_df = pd.DataFrame(X, columns=names)
            except Exception:
                X_df = pd.DataFrame(_np.asarray(X))
        else:
            X_df = pd.DataFrame(_np.asarray(X))
            if names is None and hasattr(model, "feature_names_"):
                names = list(getattr(model, "feature_names_"))
            if names is None:
                names = [f"f{i}" for i in range(X_df.shape[1])]
            X_df.columns = names

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(_np.array(X_df.values, copy=False))

        # Rank features by global impact (mean absolute SHAP)
        mean_abs = _np.abs(shap_vals).mean(axis=0)
        order = _np.argsort(-mean_abs)
        top_k = int(os.getenv("SHAP_MAX_DISPLAY", "25") or 25)
        top_k = max(10, min(top_k, X_df.shape[1]))
        top_idx = order[:top_k]

        shap_top = shap_vals[:, top_idx]
        X_top = X_df.iloc[:, top_idx]
        names_top = [X_df.columns[i] for i in top_idx]

        # 1) Beeswarm (artifact_path if provided)
        try:
            plt.close("all")
            plt.figure(figsize=(10, max(6, 0.45 * top_k)))
            shap.summary_plot(
                shap_top,
                features=X_top,
                feature_names=names_top,
                max_display=top_k,
                show=False,
            )
            bees_out = artifact_path or os.path.join(plot_dir, f"shap_beeswarm_{stem or 'model'}.png")
            plt.tight_layout()
            plt.savefig(bees_out, dpi=200)
            shap_art = bees_out
        except Exception as _be:
            logging.warning(f"SHAP beeswarm failed: {_be}")
        finally:
            plt.close()

        # 2) Bar (global importance)
        try:
            plt.close("all")
            plt.figure(figsize=(9, max(5, 0.35 * top_k)))
            shap.summary_plot(
                shap_top,
                features=X_top,
                feature_names=names_top,
                plot_type="bar",
                max_display=top_k,
                show=False,
            )
            out_name = f"shap_summary_{stem}.png" if stem else "shap_summary.png"
            out_path = os.path.join(plot_dir, out_name)
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            shap_plot = out_path
            logging.info(f"SHAP summary plot saved => {out_path}")
        except Exception as _be2:
            logging.warning(f"SHAP bar summary failed: {_be2}")
        finally:
            plt.close()
    except Exception as e:
        logging.warning(f"SHAP summary failed: {e}")
    return shap_art, shap_plot


###############################################################################
# (B) Comprehensive Correlation Heatmap (All numeric columns, all rows)
###############################################################################

def plot_full_correlation_heatmap(
    data: pd.DataFrame,
    plot_dir: str,
    filename: str = "correlation_heatmap_full.png",
    method: str = "pearson",
) -> str | None:
    """Plot a comprehensive correlation heatmap across ALL numeric columns.

    - Uses pairwise-complete observations (pandas .corr default)
    - Dynamically sizes the figure based on number of features
    - Annotates only when feature count is modest (<= 30)
    """
    try:
        # Select numeric columns only
        num_df = data.select_dtypes(include=[np.number]).copy()
        # Drop columns with no variance or all-NaN
        nunique = num_df.nunique(dropna=True)
        keep_cols = nunique[nunique > 0].index.tolist()
        num_df = num_df[keep_cols]
        if num_df.empty:
            logging.warning("Full correlation heatmap skipped: no numeric columns available.")
            return None

        corr = num_df.corr(method=method)
        n = corr.shape[0]
        # Dynamic figure size; cap extremes to keep file manageable
        fig_w = max(12, min(0.6 * n, 40))
        fig_h = max(10, min(0.5 * n, 36))

        plt.figure(figsize=(fig_w, fig_h))
        annotate = n <= 30
        sns.heatmap(corr, annot=annotate, cmap="coolwarm", center=0, vmin=-1, vmax=1, fmt=".2f" if annotate else "")
        plt.title(f"Comprehensive Correlation Heatmap ({method.title()})", fontsize=14)
        plt.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, filename)
        plt.savefig(out_path, dpi=150)
        plt.close()
        logging.info(f"Full correlation heatmap saved => {out_path}")
        return out_path
    except Exception as e:
        logging.error(f"Error in plot_full_correlation_heatmap: {e}")
        logging.error(traceback.format_exc())
        return None


###############################################################################
# (C) All-types Correlation Heatmap (includes non-numeric via safe encoding)
###############################################################################

def _encode_non_numeric_for_corr(
    df: pd.DataFrame,
    *,
    max_ohe_levels: int = 20,
) -> pd.DataFrame:
    """Return a numeric-only DataFrame by encoding non-numeric columns for correlation plotting.

    - Numeric: pass-through
    - Boolean: cast to int
    - Datetime: convert to int64 epoch (ns) scaled to days for readability
    - Object/Category/String:
      * If cardinality <= max_ohe_levels → one-hot encode
      * Else → frequency encode to a single numeric column (proportion in [0,1])
    """
    work = df.copy()
    blocks: list[pd.DataFrame] = []

    # Numeric (includes floats/ints)
    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        blocks.append(work[num_cols])

    # Booleans → int
    bool_cols = work.select_dtypes(include=[bool]).columns.tolist()
    for c in bool_cols:
        work[c] = work[c].astype(int)
    if bool_cols:
        blocks.append(work[bool_cols])

    # Datetime → epoch days
    dt_cols = work.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    for c in dt_cols:
        try:
            # nan-safe: to_numpy returns NaT as NaT; fill with nan
            vals = work[c].astype("datetime64[ns]").view("int64").astype(float)
            vals = vals / (24 * 3600 * 1e9)  # convert ns → days
            work[c + "__epoch_days"] = vals
        except Exception:
            continue
    if dt_cols:
        add_cols = [c + "__epoch_days" for c in dt_cols if (c + "__epoch_days") in work.columns]
        if add_cols:
            blocks.append(work[add_cols])

    # Object / category / string
    cat_like = work.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for c in cat_like:
        ser = work[c].astype("string").fillna("__MISSING__")
        vc = ser.value_counts(dropna=False)
        if len(vc) <= max_ohe_levels:
            dummies = pd.get_dummies(ser, prefix=c, dtype=float)
            blocks.append(dummies)
        else:
            freq = (vc / max(1, len(ser))).to_dict()
            blocks.append(pd.DataFrame({c + "__freq": ser.map(freq).astype(float)}))

    if not blocks:
        return pd.DataFrame(index=df.index)
    return pd.concat(blocks, axis=1)


def plot_correlation_heatmap_alltypes(
    data: pd.DataFrame,
    plot_dir: str,
    filename: str = "correlation_heatmap_alltypes.png",
    method: str = "pearson",
    max_ohe_levels: int = 20,
    top_k: int = 40,
    target_col: str | None = "Last Rental Price",
    cluster: bool = True,
) -> dict:
    """Plot readable, Kaggle-style correlation visualizations including non-numeric columns.

    - Encodes non-numeric for plotting only (no training impact)
    - Generates two plots by default:
        1) Top-K correlation heatmap (masked upper triangle, readable labels)
        2) Clustered heatmap (clustermap) over the same Top-K subset

    Returns a dict with paths of generated artifacts.
    """
    outputs: dict[str, str | None] = {"topk": None, "cluster": None, "full": None, "bars": None}
    try:
        enc = _encode_non_numeric_for_corr(data, max_ohe_levels=max_ohe_levels)
        if enc.empty:
            logging.warning("All-types correlation heatmap skipped: no encodable columns.")
            return outputs
        # Drop all-NaN or constant columns for stability
        enc = enc.dropna(axis=1, how="all")
        nunique = enc.nunique(dropna=False)
        enc = enc.loc[:, nunique[nunique > 1].index]
        if enc.empty:
            logging.warning("All-types correlation heatmap skipped: encoded columns are constant/NaN.")
            return outputs

        corr = enc.corr(method=method)

        # Select a readable subset of columns
        cols_sel: list[str]
        if target_col and target_col in corr.columns:
            order = corr[target_col].abs().sort_values(ascending=False)
            cols_sel = [c for c in order.index if c != target_col][: max(1, top_k)]
            cols_sel = [target_col] + cols_sel
        else:
            mean_abs = corr.abs().mean().sort_values(ascending=False)
            cols_sel = list(mean_abs.index[: max(1, top_k)])

        corr_sub = corr.loc[cols_sel, cols_sel]

        # 1) Top-K (triangle) heatmap
        n = corr_sub.shape[0]
        fig_w = max(12, min(0.42 * n + 6, 36))
        fig_h = max(10, min(0.38 * n + 5, 30))
        mask = np.triu(np.ones_like(corr_sub, dtype=bool), k=1)
        plt.figure(figsize=(fig_w, fig_h))
        annotate = n <= 25
        ax = sns.heatmap(
            corr_sub,
            mask=mask,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=False,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"shrink": 0.7},
            annot=annotate,
            fmt=".2f" if annotate else "",
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.title(f"Top-{len(cols_sel)} Correlations ({method.title()})", fontsize=14)
        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        out_topk = os.path.join(plot_dir, filename.replace(".png", "_topk.png"))
        plt.savefig(out_topk, dpi=170)
        plt.close()
        outputs["topk"] = out_topk

        # 2) Clustered heatmap for structure discovery
        if cluster and n >= 3:
            try:
                cg = sns.clustermap(
                    corr_sub,
                    cmap="coolwarm",
                    center=0,
                    vmin=-1,
                    vmax=1,
                    method="average",
                    metric="euclidean",
                    figsize=(max(10, min(0.35 * n + 6, 28)), max(10, min(0.35 * n + 6, 28))),
                    cbar_pos=(0.02, 0.8, 0.03, 0.18),
                )
                cg.fig.suptitle("Clustered Correlation (Top-K)", y=1.02, fontsize=13)
                out_cluster = os.path.join(plot_dir, filename.replace(".png", "_cluster.png"))
                cg.savefig(out_cluster, dpi=170)
                plt.close(cg.fig)
                outputs["cluster"] = out_cluster
            except Exception as e:
                logging.warning(f"Clustered heatmap failed: {e}")

        # Optional: full (unfiltered) heatmap only if small enough
        if corr.shape[0] <= 60:
            fig_w = max(12, min(0.35 * corr.shape[0] + 6, 36))
            fig_h = max(10, min(0.30 * corr.shape[0] + 5, 30))
            plt.figure(figsize=(fig_w, fig_h))
            annotate = corr.shape[0] <= 20
            sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=annotate, fmt=".2f" if annotate else "")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.title(f"All-Types Correlation (Full, {method.title()})", fontsize=14)
            plt.tight_layout()
            out_full = os.path.join(plot_dir, filename)
            plt.savefig(out_full, dpi=150)
            plt.close()
            outputs["full"] = out_full

        return outputs
    except Exception as e:
        logging.error(f"Error in plot_correlation_heatmap_alltypes: {e}")
        logging.error(traceback.format_exc())
        return outputs


def plot_average_price_per_month(data, date_col, price_col, plot_dir):
    """
    Plot and save average price per month. Possibly used in EDA.
    """
    try:
        sns.set_theme(style="whitegrid")
        if date_col not in data.columns or price_col not in data.columns:
            logging.error(f"Missing columns: {date_col}, {price_col}")
            return

        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.dropna(subset=[date_col, price_col], inplace=True)

        df['YearMonth'] = df[date_col].dt.to_period('M')
        monthly_avg = df.groupby('YearMonth')[price_col].mean().reset_index()
        monthly_avg['YearMonth'] = monthly_avg['YearMonth'].dt.to_timestamp()

        plt.figure(figsize=(10, 6))
        sns.lineplot(x='YearMonth', y=price_col, data=monthly_avg, marker='o')
        plt.xlabel('Month')
        plt.ylabel(f'Average {price_col}')
        plt.title('Average Price Per Month')
        plt.xticks(rotation=45)
        plt.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        fpath = os.path.join(plot_dir, 'average_price_per_month.png')
        plt.savefig(fpath)
        plt.close()
        logging.info(f"Saved average_price_per_month => {fpath}")

    except Exception as e:
        logging.error(f"Error in plot_average_price_per_month: {str(e)}")
        logging.error(traceback.format_exc())


def plot_cumulative_change(data, date_col, price_col, plot_dir):
    """
    Plot and save the cumulative percentage change of the average price over time.
    """
    try:
        sns.set_theme(style="whitegrid")
        if date_col not in data.columns or price_col not in data.columns:
            logging.error(f"Missing columns: {date_col}, {price_col}")
            return

        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.dropna(subset=[date_col, price_col], inplace=True)

        df['YearMonth'] = df[date_col].dt.to_period('M')
        monthly_avg = df.groupby('YearMonth')[price_col].mean().reset_index()
        monthly_avg['YearMonth'] = monthly_avg['YearMonth'].dt.to_timestamp()

        monthly_avg['Cumulative Change (%)'] = (
            (monthly_avg[price_col] / monthly_avg[price_col].iloc[0] - 1) * 100
        )

        plt.figure(figsize=(10, 6))
        sns.lineplot(x='YearMonth', y='Cumulative Change (%)', data=monthly_avg, marker='o')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel('Month')
        plt.ylabel('Cumulative Change (%)')
        plt.title('Cumulative Change of Average Price Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        fpath = os.path.join(plot_dir, 'cumulative_change.png')
        plt.savefig(fpath)
        plt.close()
        logging.info(f"Cumulative change plot => {fpath}")
    except Exception as e:
        logging.error(f"Error in plot_cumulative_change: {str(e)}")
        logging.error(traceback.format_exc())


###############################################################################
# (B) Interactive VWAP - Separate eq vs. dollar for individual bed types
###############################################################################

def plot_vwap_interactive_eq(data, bed_type, plot_dir):
    """
    Interactive eq-weighted VWAP (3M/2Y) for a single bed_type only.
    (No lines removed from your version.)
    """
    try:
        df_bed = data[data['Bed'] == bed_type].copy()
        df_bed.sort_values('Last Rental Date', inplace=True)

        needed_cols = [
            'Last Rental Date',
            'VWAP_3M_eq', 'VWAP_2Y_eq',
            'Last Rental Price', 'Volume'
        ]
        missing = [c for c in needed_cols if c not in df_bed.columns]
        if missing:
            logging.error(f"plot_vwap_interactive_eq: missing {missing} for bed={bed_type}")
            return

        # Weekly volume
        weekly_vol = (
            df_bed
            .set_index('Last Rental Date')
            .resample('W')['Volume']
            .sum()
            .reset_index()
        )

        nobs = len(df_bed)
        start_d = df_bed['Last Rental Date'].min().date()
        end_d = df_bed['Last Rental Date'].max().date()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=weekly_vol['Last Rental Date'],
            y=weekly_vol['Volume'],
            marker=dict(color='lightgray'),
            opacity=0.15,
            showlegend=False,
            name='Volume'
        ), secondary_y=True)

        fig.add_trace(go.Scatter(
            x=df_bed['Last Rental Date'],
            y=df_bed['VWAP_3M_eq'],
            mode='lines',
            name='VWAP 3M (Eq)',
            line=dict(color='blue', width=3)
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df_bed['Last Rental Date'],
            y=df_bed['VWAP_2Y_eq'],
            mode='lines',
            name='VWAP 2Y (Eq)',
            line=dict(color='red', width=3)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df_bed['Last Rental Date'],
            y=df_bed['Last Rental Price'],
            mode='markers',
            name='Actual Price',
            marker=dict(size=4, color='gray', opacity=0.6)
        ), secondary_y=False)

        title_txt = (
            f"Eq-Weighted VWAP Only\n"
            f"Bed={bed_type}, Obs={nobs}, Range={start_d} -> {end_d}"
        )
        fig.update_layout(
            title=title_txt,
            xaxis_title='Date',
            yaxis_title='Eq Weighted Price',
            xaxis=dict(rangeslider=dict(visible=True), type='date'),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            barmode='overlay',
            modebar_add=["drawline", "eraseshape"],
            newshape=dict(line_color='rgba(0,0,0,0.5)', line_dash='dot'),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        fig.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False)
        # Move the bar behind lines:
        fig.data = fig.data[::-1]

        div_id = f'plot_div_eq_{bed_type}_{id(fig)}'
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id=div_id)

        # Updated JS snippet to handle either y or y2 references:
        custom_js = f"""
        <script>
        var plot = document.getElementById('{div_id}');
        plot.on('plotly_relayout', function(eventdata){{
            if(eventdata['shapes']){{
                var shapeKeys = Object.keys(eventdata['shapes']);
                var lastShapeKey = shapeKeys[shapeKeys.length - 1];
                var shape = eventdata['shapes'][lastShapeKey];

                if(shape && shape.type === 'line'){{
                    // Check the axis references (yref might be 'y2' if drawn in the volume area)
                    var xref = shape.xref || 'x';
                    var yref = shape.yref || 'y';

                    var x0 = shape.x0; 
                    var y0 = shape.y0;
                    var x1 = shape.x1; 
                    var y1 = shape.y1;

                    var pct_change = (y0 !== 0) ? ((y1 - y0)/y0)*100 : 0;
                    var x0_date = new Date(x0).toLocaleDateString();
                    var x1_date = new Date(x1).toLocaleDateString();

                    var newAnn = plot.layout.annotations ? plot.layout.annotations.slice() : [];
                    newAnn.push(
                        {{
                            x: x0, 
                            y: y0, 
                            xref: xref, 
                            yref: yref,
                            text: 'Date: '+ x0_date +'<br>Price: '+ y0.toFixed(2),
                            showarrow: true, arrowhead: 7, ax: -40, ay: -40
                        }},
                        {{
                            x: x1, 
                            y: y1, 
                            xref: xref, 
                            yref: yref,
                            text: 'Date: '+ x1_date +'<br>Price: '+ y1.toFixed(2) +'<br>Change: '+ pct_change.toFixed(2) +'%',
                            showarrow: true, arrowhead: 7, ax: -40, ay: -40
                        }}
                    );
                    Plotly.relayout(plot, {{ 'annotations': newAnn }});
                }}
            }}
            // If user hits "erase shape"
            if(eventdata['shapes[0]'] === null){{
                Plotly.relayout(plot, {{'annotations': []}});
            }}
        }});
        </script>
        """
        html_str = html_str.replace('</body>', custom_js + '</body>')

        os.makedirs(plot_dir, exist_ok=True)
        out_fname = f'vwap_{bed_type}_eq_only.html'
        out_path = os.path.join(plot_dir, out_fname)
        with open(out_path, 'w') as ff:
            ff.write(html_str)

        logging.info(f"Eq-only interactive VWAP for bed={bed_type} => {out_path}")

    except Exception as e:
        logging.error(f"Error in plot_vwap_interactive_eq for bed={bed_type}: {e}")
        logging.error(traceback.format_exc())


def plot_vwap_interactive_dollar(data, bed_type, plot_dir):
    """
    Interactive $-weighted VWAP (3M/2Y) for a single bed_type only.
    """
    try:
        df_bed = data[data['Bed'] == bed_type].copy()
        df_bed.sort_values('Last Rental Date', inplace=True)

        needed_cols = [
            'Last Rental Date',
            'VWAP_3M', 'VWAP_2Y',
            'Last Rental Price', 'Volume'
        ]
        missing = [c for c in needed_cols if c not in df_bed.columns]
        if missing:
            logging.error(f"plot_vwap_interactive_dollar: missing {missing} for bed={bed_type}")
            return

        weekly_vol = (
            df_bed
            .set_index('Last Rental Date')
            .resample('W')['Volume']
            .sum()
            .reset_index()
        )

        nobs = len(df_bed)
        start_d = df_bed['Last Rental Date'].min().date()
        end_d = df_bed['Last Rental Date'].max().date()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=weekly_vol['Last Rental Date'],
            y=weekly_vol['Volume'],
            marker=dict(color='lightgray'),
            opacity=0.15,
            showlegend=False,
            name='Volume'
        ), secondary_y=True)

        fig.add_trace(go.Scatter(
            x=df_bed['Last Rental Date'],
            y=df_bed['VWAP_3M'],
            mode='lines',
            name='VWAP 3M ($)',
            line=dict(color='blue', width=3)
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df_bed['Last Rental Date'],
            y=df_bed['VWAP_2Y'],
            mode='lines',
            name='VWAP 2Y ($)',
            line=dict(color='red', width=3)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df_bed['Last Rental Date'],
            y=df_bed['Last Rental Price'],
            mode='markers',
            name='Actual Price',
            marker=dict(size=4, color='gray', opacity=0.6)
        ), secondary_y=False)

        title_txt = (
            f"Dollar-Weighted VWAP Only\n"
            f"Bed={bed_type}, Obs={nobs}, Range={start_d} -> {end_d}"
        )

        fig.update_layout(
            title=title_txt,
            xaxis_title='Date',
            yaxis_title='Dollar Weighted Price',
            xaxis=dict(rangeslider=dict(visible=True), type='date'),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            barmode='overlay',
            modebar_add=["drawline", "eraseshape"],
            newshape=dict(line_color='rgba(0,0,0,0.5)', line_dash='dot'),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        fig.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False)
        # Move bar behind lines
        fig.data = fig.data[::-1]

        div_id = f'plot_div_dollar_{bed_type}_{id(fig)}'
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id=div_id)

        custom_js = f"""
        <script>
        var plot = document.getElementById('{div_id}');
        plot.on('plotly_relayout', function(eventdata){{
            if(eventdata['shapes']){{
                var shapeKeys = Object.keys(eventdata['shapes']);
                var lastShapeKey = shapeKeys[shapeKeys.length - 1];
                var shape = eventdata['shapes'][lastShapeKey];
                if(shape && shape.type === 'line'){{
                    var xref = shape.xref || 'x';
                    var yref = shape.yref || 'y';

                    var x0 = shape.x0; 
                    var y0 = shape.y0;
                    var x1 = shape.x1; 
                    var y1 = shape.y1;

                    var pct_change = (y0!==0) ? ((y1 - y0)/y0)*100 : 0;
                    var x0_date = new Date(x0).toLocaleDateString();
                    var x1_date = new Date(x1).toLocaleDateString();

                    var newAnn = plot.layout.annotations ? plot.layout.annotations.slice() : [];
                    newAnn.push(
                        {{
                            x: x0, y: y0, xref: xref, yref: yref,
                            text: 'Date: '+ x0_date +'<br>Price: '+ y0.toFixed(2),
                            showarrow: true, arrowhead: 7, ax: -40, ay: -40
                        }},
                        {{
                            x: x1, y: y1, xref: xref, yref: yref,
                            text: 'Date: '+ x1_date +'<br>Price: '+ y1.toFixed(2) +'<br>Change: '+ pct_change.toFixed(2) +'%',
                            showarrow: true, arrowhead: 7, ax: -40, ay: -40
                        }}
                    );
                    Plotly.relayout(plot, {{ 'annotations': newAnn }});
                }}
            }}
            if(eventdata['shapes[0]'] === null){{
                Plotly.relayout(plot, {{'annotations': []}});
            }}
        }});
        </script>
        """
        html_str = html_str.replace('</body>', custom_js + '</body>')

        os.makedirs(plot_dir, exist_ok=True)
        out_fname = f'vwap_{bed_type}_dollar_only.html'
        out_path = os.path.join(plot_dir, out_fname)
        with open(out_path, 'w') as ff:
            ff.write(html_str)

        logging.info(f"Dollar-only interactive VWAP for bed={bed_type} => {out_path}")

    except Exception as e:
        logging.error(f"Error in plot_vwap_interactive_dollar for bed={bed_type}: {e}")
        logging.error(traceback.format_exc())


###############################################################################
# (C) Global eq/dollar - separate interactive plots
###############################################################################

def plot_global_vwap_interactive_eq(data, plot_dir):
    """
    Global eq-weighted VWAP (3M/2Y) in separate interactive chart.
    """
    try:
        df = data.copy()
        df.sort_values('Last Rental Date', inplace=True)

        needed_cols = [
            'Last Rental Date',
            'VWAP_3M_all_eq', 'VWAP_2Y_all_eq',
            'Last Rental Price', 'Volume_all'
        ]
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            logging.error(f"plot_global_vwap_interactive_eq missing: {missing}")
            return

        vol_week = (
            df
            .set_index('Last Rental Date')
            .resample('W')['Volume_all']
            .sum()
            .reset_index()
        )

        nobs = len(df)
        start_d = df['Last Rental Date'].min().date()
        end_d = df['Last Rental Date'].max().date()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=vol_week['Last Rental Date'],
            y=vol_week['Volume_all'],
            marker=dict(color='lightgray'),
            opacity=0.15,
            showlegend=False,
            name='Volume_all'
        ), secondary_y=True)

        fig.add_trace(go.Scatter(
            x=df['Last Rental Date'],
            y=df['VWAP_2Y_all_eq'],
            mode='lines',
            name='2Y VWAP (Eq All)',
            line=dict(color='firebrick', width=3)
        ), secondary_y=False)

        # Actual price scatter
        fig.add_trace(go.Scatter(
            x=df['Last Rental Date'],
            y=df['Last Rental Price'],
            mode='markers',
            name='Actual Price',
            marker=dict(size=4, color='gray', opacity=0.6)
        ), secondary_y=False)

        # Add total property count legend entry (invisible marker)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='rgba(0,0,0,0)'),
            name=f'Total Properties: {nobs}',
            showlegend=True
        ), secondary_y=False)

        title_txt = (
            f"Global EQ VWAP (2Y)\n"
            f"Obs={nobs}, Range={start_d} -> {end_d}"
        )

        fig.update_layout(
            title=title_txt,
            xaxis_title='Date',
            yaxis_title='Eq Weighted Price (All Beds)',
            xaxis=dict(rangeslider=dict(visible=True), type='date'),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            barmode='overlay',
            modebar_add=["drawline", "eraseshape"],
            newshape=dict(line_color='rgba(0,0,0,0.5)', line_dash='dot'),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        fig.update_yaxes(title_text="Volume_all", secondary_y=True, showgrid=False)
        # Move bars behind lines
        fig.data = fig.data[::-1]

        div_id = f'plot_div_global_eq_{id(fig)}'
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id=div_id)

        # Updated snippet to handle correct axis references:
        custom_js = f"""
        <script>
        var plot = document.getElementById('{div_id}');
        plot.on('plotly_relayout', function(eventdata) {{
            if(eventdata['shapes']) {{
                var shapeKeys = Object.keys(eventdata['shapes']);
                var lastShapeKey = shapeKeys[shapeKeys.length - 1];
                var shape = eventdata['shapes'][lastShapeKey];
                if(shape && shape.type === 'line') {{
                    var xref = shape.xref || 'x';
                    var yref = shape.yref || 'y';

                    var x0 = shape.x0; 
                    var y0 = shape.y0;
                    var x1 = shape.x1; 
                    var y1 = shape.y1;

                    var pct_change = (y0!==0) ? ((y1 - y0)/y0)*100 : 0;
                    var x0_date = new Date(x0).toLocaleDateString();
                    var x1_date = new Date(x1).toLocaleDateString();

                    var newAnn = plot.layout.annotations ? plot.layout.annotations.slice() : [];
                    newAnn.push(
                        {{
                            x: x0, y: y0, xref: xref, yref: yref,
                            text: 'Date: ' + x0_date + '<br>Price: ' + y0.toFixed(2),
                            showarrow: true,
                            arrowhead: 7,
                            ax: -40,
                            ay: -40
                        }},
                        {{
                            x: x1, y: y1, xref: xref, yref: yref,
                            text: 'Date: ' + x1_date + '<br>Price: ' + y1.toFixed(2) + '<br>Change: ' + pct_change.toFixed(2) + '%',
                            showarrow: true,
                            arrowhead: 7,
                            ax: -40,
                            ay: -40
                        }}
                    );

                    Plotly.relayout(plot, {{ 'annotations': newAnn }});
                }}
            }}
            if(eventdata['shapes[0]'] === null) {{
                Plotly.relayout(plot, {{'annotations': []}});
            }}
        }});
        </script>
        """
        html_str = html_str.replace('</body>', custom_js + '</body>')

        os.makedirs(plot_dir, exist_ok=True)
        out_name = 'vwap_all_beds_eq_only.html'
        out_path = os.path.join(plot_dir, out_name)
        with open(out_path, 'w') as f:
            f.write(html_str)

        logging.info(f"Global eq-only interactive VWAP => {out_path}")

    except Exception as e:
        logging.error(f"Error in plot_global_vwap_interactive_eq: {e}")
        logging.error(traceback.format_exc())


def plot_global_vwap_interactive_dollar(data, plot_dir):
    """
    Global $-weighted VWAP (3M/2Y) in a separate interactive chart.
    """
    try:
        df = data.copy()
        df.sort_values('Last Rental Date', inplace=True)

        needed_cols = [
            'Last Rental Date',
            'VWAP_3M_all', 'VWAP_2Y_all',
            'Last Rental Price', 'Volume_all'
        ]
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            logging.error(f"plot_global_vwap_interactive_dollar missing: {missing}")
            return

        vol_week = (
            df
            .set_index('Last Rental Date')
            .resample('W')['Volume_all']
            .sum()
            .reset_index()
        )

        nobs = len(df)
        start_d = df['Last Rental Date'].min().date()
        end_d = df['Last Rental Date'].max().date()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=vol_week['Last Rental Date'],
            y=vol_week['Volume_all'],
            marker=dict(color='lightgray'),
            opacity=0.15,
            showlegend=False,
            name='Volume_all'
        ), secondary_y=True)

        fig.add_trace(go.Scatter(
            x=df['Last Rental Date'],
            y=df['VWAP_2Y_all'],
            mode='lines',
            name='2Y VWAP (Dollar All)',
            line=dict(color='firebrick', width=3)
        ), secondary_y=False)

        # Actual price scatter
        fig.add_trace(go.Scatter(
            x=df['Last Rental Date'],
            y=df['Last Rental Price'],
            mode='markers',
            name='Actual Price',
            marker=dict(size=4, color='gray', opacity=0.6)
        ), secondary_y=False)

        # Add total property count legend entry (invisible marker)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='rgba(0,0,0,0)'),
            name=f'Total Properties: {nobs}',
            showlegend=True
        ), secondary_y=False)

        title_txt = (
            f"Global Dollar-Weighted VWAP Only (2Y)\n"
            f"Obs={nobs}, Range={start_d} -> {end_d}"
        )

        fig.update_layout(
            title=title_txt,
            xaxis_title='Date',
            yaxis_title='Dollar Weighted Price (All Beds)',
            xaxis=dict(rangeslider=dict(visible=True), type='date'),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            barmode='overlay',
            modebar_add=["drawline", "eraseshape"],
            newshape=dict(line_color='rgba(0,0,0,0.5)', line_dash='dot'),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        fig.update_yaxes(title_text="Volume_all", secondary_y=True, showgrid=False)
        # Move bars behind lines
        fig.data = fig.data[::-1]

        div_id = f'plot_div_global_dollar_{id(fig)}'
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id=div_id)

        custom_js = f"""
        <script>
        var plot = document.getElementById('{div_id}');
        plot.on('plotly_relayout', function(eventdata){{
            if(eventdata['shapes']){{
                var shapeKeys = Object.keys(eventdata['shapes']);
                var lastShapeKey = shapeKeys[shapeKeys.length - 1];
                var shape = eventdata['shapes'][lastShapeKey];
                if(shape && shape.type === 'line'){{
                    var xref = shape.xref || 'x';
                    var yref = shape.yref || 'y';

                    var x0 = shape.x0; 
                    var y0 = shape.y0;
                    var x1 = shape.x1; 
                    var y1 = shape.y1;

                    var pct_change = (y0!==0) ? ((y1 - y0)/y0)*100 : 0;
                    var x0_date = new Date(x0).toLocaleDateString();
                    var x1_date = new Date(x1).toLocaleDateString();

                    var newAnn = plot.layout.annotations ? plot.layout.annotations.slice() : [];
                    newAnn.push(
                        {{
                            x: x0, y: y0, xref: xref, yref: yref,
                            text: 'Date: '+ x0_date +'<br>Price: '+ y0.toFixed(2),
                            showarrow: true, arrowhead: 7, ax: -40, ay: -40
                        }},
                        {{
                            x: x1, y: y1, xref: xref, yref: yref,
                            text: 'Date: '+ x1_date +'<br>Price: '+ y1.toFixed(2) +'<br>Change: '+ pct_change.toFixed(2) +'%',
                            showarrow: true, arrowhead: 7, ax: -40, ay: -40
                        }}
                    );
                    Plotly.relayout(plot, {{ 'annotations': newAnn }});
                }}
            }}
            if(eventdata['shapes[0]'] === null){{
                Plotly.relayout(plot, {{'annotations': []}});
            }}
        }});
        </script>
        """
        html_str = html_str.replace('</body>', custom_js + '</body>')

        os.makedirs(plot_dir, exist_ok=True)
        out_name = 'vwap_all_beds_dollar_only.html'
        out_path = os.path.join(plot_dir, out_name)
        with open(out_path, 'w') as f:
            f.write(html_str)

        logging.info(f"Global dollar-only interactive VWAP => {out_path}")

    except Exception as e:
        logging.error(f"Error in plot_global_vwap_interactive_dollar: {e}")
        logging.error(traceback.format_exc())


###############################################################################
# (D) Interactive Median Plot for Per-Bed Type (NEW)
###############################################################################

def plot_bed_median_interactive(data, bed_type, plot_dir):
    """
    Plots INTERACTIVE 3M & 2Y Rolling Medians for a specific bed type.
    Uses 'Rolling_Median_3M_bed' and 'Rolling_Median_2Y_bed'.
    """
    try:
        df_bed = data[data['Bed'] == bed_type].copy()
        df_bed.sort_values('Last Rental Date', inplace=True)

        median_3m_col = 'Rolling_Median_3M_bed'
        median_2y_col = 'Rolling_Median_2Y_bed'
        price_col = 'Last Rental Price'
        date_col = 'Last Rental Date'
        volume_col = 'Volume' 

        needed_cols = [date_col, median_3m_col, median_2y_col, price_col]
        if volume_col not in df_bed.columns:
            logging.info(f"'{volume_col}' not found for bed type {bed_type}, will not be plotted.")
        else:
            needed_cols.append(volume_col)

        missing = [c for c in needed_cols if c not in df_bed.columns]
        if missing:
            if median_3m_col not in df_bed.columns or median_2y_col not in df_bed.columns:
                logging.error(f"plot_bed_median_interactive: Essential median columns ({median_3m_col}, {median_2y_col}) missing for bed={bed_type}. Cannot create plot.")
                return
            logging.warning(f"plot_bed_median_interactive: Some columns missing {missing} for bed={bed_type}. Proceeding with available data.")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        if volume_col in df_bed.columns:
            weekly_vol = (
                df_bed
                .set_index(date_col)
                .resample('W')[volume_col]
                .sum()
                .reset_index()
            )
            fig.add_trace(go.Bar(
                x=weekly_vol[date_col],
                y=weekly_vol[volume_col],
                marker=dict(color='lightgray'),
                opacity=0.15,
                showlegend=False,
                name='Volume'
            ), secondary_y=True)
            fig.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False)
            if fig.data:
                 fig.data = fig.data[::-1]

        if median_3m_col in df_bed.columns:
            fig.add_trace(go.Scatter(
                x=df_bed[date_col],
                y=df_bed[median_3m_col],
                mode='lines',
                name='3M Rolling Median (Bed)',
                line=dict(color='green', width=3)
            ), secondary_y=False)
        
        if median_2y_col in df_bed.columns:
            fig.add_trace(go.Scatter(
                x=df_bed[date_col],
                y=df_bed[median_2y_col],
                mode='lines',
                name='2Y Rolling Median (Bed)',
                line=dict(color='purple', width=3)
            ), secondary_y=False)

        if price_col in df_bed.columns:
            fig.add_trace(go.Scatter(
                x=df_bed[date_col],
                y=df_bed[price_col],
                mode='markers',
                name='Actual Price',
                marker=dict(size=4, color='gray', opacity=0.6)
            ), secondary_y=False)

        nobs = len(df_bed)
        start_d = df_bed[date_col].min().date() if not df_bed[date_col].empty else 'N/A'
        end_d = df_bed[date_col].max().date() if not df_bed[date_col].empty else 'N/A'
        title_txt = (
            f"Interactive Rolling Medians for Bed={bed_type}\n"
            f"Obs={nobs}, Range={start_d} -> {end_d}"
        )
        fig.update_layout(
            title=title_txt,
            xaxis_title='Date',
            yaxis_title='Median Rental Price (Per Bed)',
            xaxis=dict(rangeslider=dict(visible=True), type='date'),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            barmode='overlay',
            modebar_add=["drawline", "eraseshape"],
            newshape=dict(line_color='rgba(0,0,0,0.5)', line_dash='dot'),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        div_id = f'plot_div_bed_median_{bed_type}_{id(fig)}'
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id=div_id)

        custom_js = f'''
        <script>
        var plot_element = document.getElementById('{div_id}');
        if (plot_element) {{
            plot_element.on('plotly_relayout', function(eventdata){{
                if(eventdata['shapes']){{
                    var shapeKeys = Object.keys(eventdata['shapes']);
                    var lastShapeKey = shapeKeys[shapeKeys.length - 1];
                    var shape = eventdata['shapes'][lastShapeKey];
                    if(shape && shape.type === 'line'){{                        
                        var xref = shape.xref || 'x';
                        var yref = shape.yref || 'y';
                        var x0 = shape.x0; var y0 = shape.y0;
                        var x1 = shape.x1; var y1 = shape.y1;
                        var pct_change = (y0!==0) ? ((y1 - y0)/y0)*100 : 0;
                        var x0_date = new Date(x0).toLocaleDateString();
                        var x1_date = new Date(x1).toLocaleDateString();
                        var newAnn = plot_element.layout.annotations ? plot_element.layout.annotations.slice() : [];
                        newAnn.push(
                            {{x: x0, y: y0, xref: xref, yref: yref, text: 'Date: '+ x0_date +'<br>Price: '+ y0.toFixed(2), showarrow: true, arrowhead: 7, ax: -40, ay: -40}},
                            {{x: x1, y: y1, xref: xref, yref: yref, text: 'Date: '+ x1_date +'<br>Price: '+ y1.toFixed(2) +'<br>Change: '+ pct_change.toFixed(2) +'%', showarrow: true, arrowhead: 7, ax: -40, ay: -40}}
                        );
                        Plotly.relayout(plot_element, {{ 'annotations': newAnn }});
                    }}
                }}
                if(eventdata['shapes[0]'] === null){{ Plotly.relayout(plot_element, {{'annotations': []}}); }}
            }});
        }}
        </script>
        '''
        html_str = html_str.replace('</body>', custom_js + '</body>')

        os.makedirs(plot_dir, exist_ok=True)
        out_fname = f'median_bed_{bed_type}_interactive.html'
        out_path = os.path.join(plot_dir, out_fname)
        with open(out_path, 'w') as ff:
            ff.write(html_str)

        logging.info(f"Per-bed interactive Rolling Median plot for bed={bed_type} => {out_path}")

    except Exception as e:
        logging.error(f"Error in plot_bed_median_interactive for bed={bed_type}: {e}")
        logging.error(traceback.format_exc())


###############################################################################
# (E) Fallback or legacy calls - REVISED
###############################################################################

def plot_vwap_interactive(data, bed_type, plot_dir):
    """ 
    Main interactive plotter for a bed type.
    PRIORITIZES Per-Bed Rolling Medians.
    Falls back to Equal-Weighted VWAP, then Dollar-Weighted VWAP if medians/eq_vwap are missing.
    """
    median_3m_col = 'Rolling_Median_3M_bed'
    median_2y_col = 'Rolling_Median_2Y_bed'
    vwap_3m_eq_col = 'VWAP_3M_eq'
    vwap_2y_eq_col = 'VWAP_2Y_eq'
    vwap_3m_dollar_col = 'VWAP_3M'
    vwap_2y_dollar_col = 'VWAP_2Y'

    bed_data = data[data['Bed'] == bed_type]

    if median_3m_col in bed_data.columns and median_2y_col in bed_data.columns and \
       not bed_data[[median_3m_col, median_2y_col]].isnull().all().all():
        logging.info(f"Using PER-BED ROLLING MEDIANS for interactive plot for bed={bed_type}.")
        return plot_bed_median_interactive(data, bed_type, plot_dir)
    elif vwap_3m_eq_col in bed_data.columns and vwap_2y_eq_col in bed_data.columns and \
       not bed_data[[vwap_3m_eq_col, vwap_2y_eq_col]].isnull().all().all():
        logging.warning(f"plot_vwap_interactive: Per-bed medians not available for bed={bed_type}. Falling back to EQUAL-WEIGHTED VWAP.")
        return plot_vwap_interactive_eq(data, bed_type, plot_dir)
    elif vwap_3m_dollar_col in bed_data.columns and vwap_2y_dollar_col in bed_data.columns and \
         not bed_data[[vwap_3m_dollar_col, vwap_2y_dollar_col]].isnull().all().all():
        logging.warning(f"plot_vwap_interactive: Per-bed medians AND Equal-weighted VWAP not available for bed={bed_type}. Falling back to DOLLAR-WEIGHTED VWAP.")
        return plot_vwap_interactive_dollar(data, bed_type, plot_dir)
    else:
        logging.error(f"plot_vwap_interactive: No suitable median or VWAP data available for bed={bed_type}. Cannot generate plot.")
        return None 

def plot_global_vwap_interactive(data, plot_dir):
    """
    Main interactive GLOBAL plotter.
    PRIORITIZES Global Rolling Medians (using existing plot_global_median_interactive).
    Falls back to Global Equal-Weighted VWAP, then Global Dollar-Weighted VWAP.
    """
    median_1m_global_col = 'Rolling_Median_1M_all'
    median_2y_global_col = 'Rolling_Median_2Y_all'
    vwap_3m_all_eq_col = 'VWAP_3M_all_eq'
    vwap_2y_all_eq_col = 'VWAP_2Y_all_eq'
    vwap_3m_all_dollar_col = 'VWAP_3M_all'
    vwap_2y_all_dollar_col = 'VWAP_2Y_all'

    if median_1m_global_col in data.columns and median_2y_global_col in data.columns and \
        not data[[median_1m_global_col, median_2y_global_col]].isnull().all().all():
        logging.info("Using GLOBAL ROLLING MEDIANS for interactive plot (via plot_global_median_interactive).")
        return plot_global_median_interactive(data, plot_dir) 
    elif vwap_3m_all_eq_col in data.columns and vwap_2y_all_eq_col in data.columns and \
       not data[[vwap_3m_all_eq_col, vwap_2y_all_eq_col]].isnull().all().all():
        logging.info("Global medians not suitable/available through this wrapper, using GLOBAL EQUAL-WEIGHTED VWAP for interactive plot.")
        return plot_global_vwap_interactive_eq(data, plot_dir)
    elif vwap_3m_all_dollar_col in data.columns and vwap_2y_all_dollar_col in data.columns and \
         not data[[vwap_3m_all_dollar_col, vwap_2y_all_dollar_col]].isnull().all().all():
        logging.warning("plot_global_vwap_interactive: Global medians and Global equal-weighted VWAP not available. Falling back to GLOBAL DOLLAR-WEIGHTED.")
        return plot_global_vwap_interactive_dollar(data, plot_dir)
    else:
        logging.error("plot_global_vwap_interactive: No suitable global median or VWAP data available. Cannot generate plot.")
        return None

###############################################################################
# (F) Global median interactive, to avoid import issues - Renamed from (E)
###############################################################################

def plot_global_median_interactive(data, plot_dir):
    """
    Interactive global rolling median chart for all-beds combined.
    """
    try:
        median_data = data.copy()
        median_data.sort_values('Last Rental Date', inplace=True)

        required_cols = [
            'Last Rental Date',
            'Rolling_Median_2Y_all',
            'Last Rental Price'
        ]
        missing_cols = [c for c in required_cols if c not in median_data.columns]
        if missing_cols:
            # If the 2Y median (which we *are* plotting) is missing, then it's an issue.
            if 'Rolling_Median_2Y_all' not in median_data.columns:
                 logging.error(f"Missing required column for plotting: Rolling_Median_2Y_all")
                 return
            logging.warning(f"Missing some columns in median_data: {missing_cols}, but proceeding with 2Y median.")

        num_obs = len(median_data)
        start_date = median_data['Last Rental Date'].min().date()
        end_date = median_data['Last Rental Date'].max().date()

        title_text = (
            f"Interactive Global 2Y Rolling Median (All Beds)\n"
            f"Observations: {num_obs}, Period: {start_date} - {end_date}"
        )

        fig = go.Figure()

        # 2Y Rolling Median
        fig.add_trace(go.Scatter(
            x=median_data['Last Rental Date'],
            y=median_data['Rolling_Median_2Y_all'],
            mode='lines',
            name='2Y Rolling Median',
            line=dict(color='red', width=3)
        ))

        # Actual Price
        fig.add_trace(go.Scatter(
            x=median_data['Last Rental Date'],
            y=median_data['Last Rental Price'],
            mode='markers',
            name='Actual Price',
            marker=dict(size=4, color='gray', opacity=0.6)
        ))

        fig.update_layout(
            title=title_text,
            xaxis_title='Date',
            yaxis_title='Rental Price',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            ),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=40, r=40, t=80, b=40),
            modebar_add=["drawline", "eraseshape"],
            newshape=dict(line_color='rgba(0,0,0,0.5)', line_dash='dot')
        )

        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = 'rolling_median_all_beds_interactive.html'
        plot_path = os.path.join(plot_dir, plot_filename)
        
        # Generate HTML and add custom JS for annotations
        div_id = f'plot_div_global_median_{id(fig)}'
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id=div_id)

        custom_js = f'''
        <script>
        var plot_element = document.getElementById('{div_id}');
        if (plot_element) {{
            plot_element.on('plotly_relayout', function(eventdata){{
                if(eventdata['shapes']){{
                    var shapeKeys = Object.keys(eventdata['shapes']);
                    var lastShapeKey = shapeKeys[shapeKeys.length - 1];
                    var shape = eventdata['shapes'][lastShapeKey];
                    if(shape && shape.type === 'line'){{                        
                        var xref = shape.xref || 'x';
                        var yref = shape.yref || 'y';
                        var x0 = shape.x0; var y0 = shape.y0;
                        var x1 = shape.x1; var y1 = shape.y1;
                        var pct_change = (y0!==0) ? ((y1 - y0)/y0)*100 : 0;
                        var x0_date = new Date(x0).toLocaleDateString();
                        var x1_date = new Date(x1).toLocaleDateString();
                        var newAnn = plot_element.layout.annotations ? plot_element.layout.annotations.slice() : [];
                        newAnn.push(
                            {{x: x0, y: y0, xref: xref, yref: yref, text: 'Date: '+ x0_date +'<br>Price: '+ y0.toFixed(2), showarrow: true, arrowhead: 7, ax: -40, ay: -40}},
                            {{x: x1, y: y1, xref: xref, yref: yref, text: 'Date: '+ x1_date +'<br>Price: '+ y1.toFixed(2) +'<br>Change: '+ pct_change.toFixed(2) +'%', showarrow: true, arrowhead: 7, ax: -40, ay: -40}}
                        );
                        Plotly.relayout(plot_element, {{ 'annotations': newAnn }});
                    }}
                }}
                if(eventdata['shapes[0]'] === null){{ Plotly.relayout(plot_element, {{'annotations': []}}); }}
            }});
        }}
        </script>
        '''
        html_str = html_str.replace('</body>', custom_js + '</body>')

        with open(plot_path, 'w') as ff:
            ff.write(html_str)
            
        logging.info(f"plot_global_median_interactive => {plot_path}")

    except Exception as e:
        logging.error(f"Error in plot_global_median_interactive: {e}")
        logging.error(traceback.format_exc())


###############################################################################
# (G) NEW: Combined interactive eq plot for 1 & 2 bed properties
###############################################################################

def plot_eq_interactive_beds_1_and_2(data: pd.DataFrame, plot_dir: str):
    """
    Creates a single combined eq-weighted VWAP plot for Bed=1 and Bed=2.
    Aggregates bed=1 + bed=2 rows by date, then computes eq VWAP_3M & VWAP_2Y.
    Plots them on one chart with volume bars and actual prices.
    """
    try:
        combined = data[data['Bed'].isin([1,2])].copy()
        if combined.empty:
            logging.warning("No data for bed=1 or bed=2. Cannot plot combined eq.")
            return

        combined.sort_values('Last Rental Date', inplace=True)

        # Prepare daily aggregates
        combined['eq_volume'] = 1.0 # Equal weighting for each transaction
        daily_agg = (
            combined.groupby('Last Rental Date', as_index=False)
            .agg(
                sum_price=('Last Rental Price', 'sum'),
                eq_volume_sum=('eq_volume', 'sum') # Sum of transactions for the day
            )
        )
        
        daily_agg.set_index('Last Rental Date', inplace=True)
        daily_agg.sort_index(inplace=True)

        # Rolling 3M eq VWAP
        daily_agg['Rolling_Sum_Price_3M'] = daily_agg['sum_price'].rolling('90D', min_periods=3).sum()
        daily_agg['Rolling_Eq_Volume_3M']  = daily_agg['eq_volume_sum'].rolling('90D', min_periods=3).sum()
        daily_agg['VWAP_3M_eq_combined'] = (daily_agg['Rolling_Sum_Price_3M'] / daily_agg['Rolling_Eq_Volume_3M']).shift(1)

        # Rolling 2Y eq VWAP
        daily_agg['Rolling_Sum_Price_2Y'] = daily_agg['sum_price'].rolling('730D', min_periods=1).sum()
        daily_agg['Rolling_Eq_Volume_2Y']  = daily_agg['eq_volume_sum'].rolling('730D', min_periods=1).sum()
        daily_agg['VWAP_2Y_eq_combined'] = (daily_agg['Rolling_Sum_Price_2Y'] / daily_agg['Rolling_Eq_Volume_2Y']).shift(1)

        daily_agg.reset_index(inplace=True)
        
        # For plotting actuals, we can use the original combined data or daily average actuals
        # Let's use daily average actuals for clarity on the plot
        daily_actual_avg_price = (
            combined.groupby('Last Rental Date', as_index=False)['Last Rental Price'].mean()
            .rename(columns={'Last Rental Price': 'avg_actual_price_daily'})
        )
        merged_plot_data = pd.merge(daily_agg, daily_actual_avg_price, on='Last Rental Date', how='left')

        nobs = len(combined) 
        min_date = merged_plot_data['Last Rental Date'].min().date() if not merged_plot_data['Last Rental Date'].empty else 'N/A'
        max_date = merged_plot_data['Last Rental Date'].max().date() if not merged_plot_data['Last Rental Date'].empty else 'N/A'
        title_txt = (
            f"Combined EQ-Weighted VWAP for Bed=1 & Bed=2\n"
            f"Total Obs (1&2 Bed): {nobs}, Date Range: {min_date} => {max_date}"
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Volume bars (sum of daily transactions for 1&2 bed)
        fig.add_trace(go.Bar(
            x=merged_plot_data['Last Rental Date'],
            y=merged_plot_data['eq_volume_sum'],
            marker=dict(color='lightgray'),
            opacity=0.2,
            showlegend=False,
            name='Combined Daily Transactions (1&2 Bed)'
        ), secondary_y=True)

        # 3M Combined EQ VWAP
        fig.add_trace(go.Scatter(
            x=merged_plot_data['Last Rental Date'],
            y=merged_plot_data['VWAP_3M_eq_combined'],
            mode='lines',
            name='Bed1+2 VWAP_3M_eq',
            line=dict(color='blue', width=2)
        ), secondary_y=False)

        # 2Y Combined EQ VWAP
        fig.add_trace(go.Scatter(
            x=merged_plot_data['Last Rental Date'],
            y=merged_plot_data['VWAP_2Y_eq_combined'],
            mode='lines',
            name='Bed1+2 VWAP_2Y_eq',
            line=dict(color='red', width=2)
        ), secondary_y=False)

        # Daily Average Actual Price for 1&2 Bed
        fig.add_trace(go.Scatter(
            x=merged_plot_data['Last Rental Date'],
            y=merged_plot_data['avg_actual_price_daily'],
            mode='markers',
            name='Avg Actual Price (Daily, 1&2 Bed)',
            marker=dict(size=5, color='gray', opacity=0.5)
        ), secondary_y=False)

        fig.update_layout(
            title=title_txt,
            xaxis_title='Date',
            yaxis_title='Rental Price (Beds 1+2 Combined EQ VWAP)',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            ),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            barmode='overlay',
            modebar_add=["drawline", "eraseshape"],
            newshape=dict(line_color='rgba(0,0,0,0.5)', line_dash='dot'),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        fig.update_yaxes(title_text="Combined Daily Transactions (1&2 Bed)", secondary_y=True, showgrid=False)
        if fig.data: # Ensure data exists before trying to reverse
            fig.data = fig.data[::-1] # Move bars behind lines

        div_id = f"plot_div_eq_1and2_combined_{id(fig)}"
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id=div_id)

        custom_js = f'''
        <script>
        var plot_element = document.getElementById('{div_id}');
        if (plot_element) {{
            plot_element.on('plotly_relayout', function(eventdata){{
                if(eventdata['shapes']){{
                    var shapeKeys = Object.keys(eventdata['shapes']);
                    var lastShapeKey = shapeKeys[shapeKeys.length - 1];
                    var shape = eventdata['shapes'][lastShapeKey];
                    if(shape && shape.type === 'line'){{                        
                        var xref = shape.xref || 'x';
                        var yref = shape.yref || 'y';
                        var x0 = shape.x0; var y0 = shape.y0;
                        var x1 = shape.x1; var y1 = shape.y1;
                        var pct_change = (y0!==0) ? ((y1 - y0)/y0)*100 : 0;
                        var x0_date = new Date(x0).toLocaleDateString();
                        var x1_date = new Date(x1).toLocaleDateString();
                        var newAnn = plot_element.layout.annotations ? plot_element.layout.annotations.slice() : [];
                        newAnn.push(
                            {{x: x0, y: y0, xref: xref, yref: yref, text: 'Date: '+ x0_date +'<br>Price: '+ y0.toFixed(2), showarrow: true, arrowhead: 7, ax: -40, ay: -40}},
                            {{x: x1, y: y1, xref: xref, yref: yref, text: 'Date: '+ x1_date +'<br>Price: '+ y1.toFixed(2) +'<br>Change: '+ pct_change.toFixed(2) +'%', showarrow: true, arrowhead: 7, ax: -40, ay: -40}}
                        );
                        Plotly.relayout(plot_element, {{ 'annotations': newAnn }});
                    }}
                }}
                if(eventdata['shapes[0]'] === null){{ Plotly.relayout(plot_element, {{'annotations': []}}); }}
            }});
        }}
        </script>
        '''
        html_str = html_str.replace('</body>', custom_js + '</body>')

        os.makedirs(plot_dir, exist_ok=True)
        out_name = "eq_beds_1and2_combined_interactive.html"
        out_path = os.path.join(plot_dir, out_name)
        with open(out_path, 'w') as f:
            f.write(html_str)

        logging.info(f"Combined eq VWAP chart for bed=1 & 2 => {out_path}")

    except Exception as e:
        logging.error(f"Error in plot_eq_interactive_beds_1_and_2: {e}")
        logging.error(traceback.format_exc())

# ------------------------------------------------------------------
# Global flag to enable / disable heavyweight VWAP & per-bed interactive
# plots (vwap_0.0bed, median_bed_0.0_interactive, etc.).  Default=False.
# Activate by setting env variable ENABLE_VWAP_PLOTS=true
# ------------------------------------------------------------------
import os as _os

ENABLE_VWAP_PLOTS: bool = _os.getenv("ENABLE_VWAP_PLOTS", "False").strip().lower() in {"1", "true", "yes"}

# If disabled, replace VWAP plotting functions with no-ops so downstream
# calls succeed silently without generating heavy Plotly HTML files.

if not ENABLE_VWAP_PLOTS:
    def _noop(*args, **kwargs):
        logging.info("VWAP plots disabled via ENABLE_VWAP_PLOTS flag – skipping plot generation.")

    # Bed-level VWAP plots
    plot_vwap_interactive_eq = _noop  # type: ignore
    plot_vwap_interactive_dollar = _noop  # type: ignore
    plot_vwap_interactive = _noop  # type: ignore

    # Global VWAP plots
    plot_global_vwap_interactive_eq = _noop  # type: ignore
    plot_global_vwap_interactive_dollar = _noop  # type: ignore
    plot_global_vwap_interactive = _noop  # type: ignore

    # Median / rolling median helpers that rely on VWAP data
    plot_global_median_interactive = _noop  # type: ignore
    plot_eq_interactive_beds_1_and_2 = _noop  # type: ignore

    logging.info("All VWAP-related plotting functions have been stubbed out (ENABLE_VWAP_PLOTS=False).")
def save_cleaned_full_dataset(df: pd.DataFrame, plot_dir: str, stem: str | None = None) -> str | None:
    try:
        os.makedirs(plot_dir, exist_ok=True)
        name = f"cleaned_full_{stem}.csv" if stem else "cleaned_full.csv"
        path = os.path.join(plot_dir, name)
        df.to_csv(path, index=False)
        logging.info(f"Saved full cleaned dataset => {path}")
        return path
    except Exception as e:
        logging.warning(f"Saving full cleaned dataset failed: {e}")
        return None


def save_dataframe_csv(df: pd.DataFrame, plot_dir: str, filename: str) -> str | None:
    try:
        os.makedirs(plot_dir, exist_ok=True)
        path = os.path.join(plot_dir, filename)
        df.to_csv(path, index=False)
        logging.info(f"Saved CSV => {path}")
        return path
    except Exception as e:
        logging.warning(f"Saving CSV failed: {e}")
        return None
