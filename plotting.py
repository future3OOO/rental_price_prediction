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


def plot_actual_vs_predicted(y_true, y_pred, plot_dir):
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
        fpath = os.path.join(plot_dir, 'actual_vs_predicted.png')
        plt.savefig(fpath)
        plt.close(fig)
        logging.info(f"Actual vs Predicted plot saved => {fpath}")
    except Exception as e:
        logging.error(f"Error in plot_actual_vs_predicted: {str(e)}")
        logging.error(traceback.format_exc())


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
    Interactive eq-weighted VWAP (3M/12M) for a single bed_type only.
    """
    try:
        df_bed = data[data['Bed'] == bed_type].copy()
        df_bed.sort_values('Last Rental Date', inplace=True)

        needed_cols = [
            'Last Rental Date',
            'VWAP_3M_eq', 'VWAP_12M_eq',
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
            y=df_bed['VWAP_12M_eq'],
            mode='lines',
            name='VWAP 12M (Eq)',
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
    Interactive $-weighted VWAP (3M/12M) for a single bed_type only.
    """
    try:
        df_bed = data[data['Bed'] == bed_type].copy()
        df_bed.sort_values('Last Rental Date', inplace=True)

        needed_cols = [
            'Last Rental Date',
            'VWAP_3M', 'VWAP_12M',
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
            y=df_bed['VWAP_12M'],
            mode='lines',
            name='VWAP 12M ($)',
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

        # Same improved JS snippet:
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
    Global eq-weighted VWAP (3M/12M) in separate interactive chart.
    **Updated** to handle secondary axis shape-drawing for % change annotation.
    """
    try:
        df = data.copy()
        df.sort_values('Last Rental Date', inplace=True)

        needed_cols = [
            'Last Rental Date',
            'VWAP_3M_all_eq', 'VWAP_12M_all_eq',
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
            y=df['VWAP_3M_all_eq'],
            mode='lines',
            name='3M VWAP (Eq All)',
            line=dict(color='blue', width=3)
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df['Last Rental Date'],
            y=df['VWAP_12M_all_eq'],
            mode='lines',
            name='12M VWAP (Eq All)',
            line=dict(color='red', width=3)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['Last Rental Date'],
            y=df['Last Rental Price'],
            mode='markers',
            name='Actual Price',
            marker=dict(size=4, color='gray', opacity=0.6)
        ), secondary_y=False)

        title_txt = (
            f"Global EQ Weighted VWAP Only (3M/12M)\n"
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
    Global $-weighted VWAP (3M/12M) in a separate interactive chart.
    """
    try:
        df = data.copy()
        df.sort_values('Last Rental Date', inplace=True)

        needed_cols = [
            'Last Rental Date',
            'VWAP_3M_all', 'VWAP_12M_all',
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
            y=df['VWAP_3M_all'],
            mode='lines',
            name='3M VWAP ($ All)',
            line=dict(color='blue', width=3)
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df['Last Rental Date'],
            y=df['VWAP_12M_all'],
            mode='lines',
            name='12M VWAP ($ All)',
            line=dict(color='red', width=3)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['Last Rental Date'],
            y=df['Last Rental Price'],
            mode='markers',
            name='Actual Price',
            marker=dict(size=4, color='gray', opacity=0.6)
        ), secondary_y=False)

        title_txt = (
            f"Global Dollar-Weighted VWAP Only (3M/12M)\n"
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
# (D) Fallback or legacy calls
###############################################################################

def plot_vwap_interactive(data, bed_type, plot_dir):
    """
    Fallback if 'plot_vwap_interactive' is called. 
    Default to eq-only for the single bed type.
    """
    logging.info(f"plot_vwap_interactive fallback => eq-only for bed={bed_type}")
    plot_vwap_interactive_eq(data, bed_type, plot_dir)


def plot_global_vwap_interactive(data, plot_dir):
    """
    Fallback if 'plot_global_vwap_interactive' is called. 
    Default to eq-only for all-beds.
    """
    logging.info("plot_global_vwap_interactive fallback => eq-only (all-beds).")
    plot_global_vwap_interactive_eq(data, plot_dir)


###############################################################################
# (E) Global median interactive, to avoid import issues
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
            'Rolling_Median_1M_all',
            'Rolling_Median_12M_all',
            'Last Rental Price'
        ]
        missing_cols = [c for c in required_cols if c not in median_data.columns]
        if missing_cols:
            logging.error(f"Missing required columns in median_data: {missing_cols}")
            return

        num_obs = len(median_data)
        start_date = median_data['Last Rental Date'].min().date()
        end_date = median_data['Last Rental Date'].max().date()

        title_text = (
            f"Interactive Global Rolling Medians (All Beds)\n"
            f"Observations: {num_obs}, Period: {start_date} - {end_date}"
        )

        fig = go.Figure()

        # 1M Rolling Median
        fig.add_trace(go.Scatter(
            x=median_data['Last Rental Date'],
            y=median_data['Rolling_Median_1M_all'],
            mode='lines',
            name='1M Rolling Median',
            line=dict(color='blue', width=3)
        ))

        # 12M Rolling Median
        fig.add_trace(go.Scatter(
            x=median_data['Last Rental Date'],
            y=median_data['Rolling_Median_12M_all'],
            mode='lines',
            name='12M Rolling Median',
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
            margin=dict(l=40, r=40, t=80, b=40)
        )

        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = 'rolling_median_all_beds_interactive.html'
        plot_path = os.path.join(plot_dir, plot_filename)
        fig.write_html(plot_path)
        logging.info(f"plot_global_median_interactive => {plot_path}")

    except Exception as e:
        logging.error(f"Error in plot_global_median_interactive: {e}")
        logging.error(traceback.format_exc())


###############################################################################
# (F) NEW: Combined interactive eq plot for 1 & 2 bed properties
###############################################################################

def plot_eq_interactive_beds_1_and_2(data: pd.DataFrame, plot_dir: str):
    """
    Creates a single combined eq-weighted VWAP plot for Bed=1 and Bed=2.
    Aggregates bed=1 + bed=2 rows by date, then computes eq VWAP_3M & VWAP_12M.
    Plots them on one chart with volume bars and actual prices.
    """
    try:
        combined = data[data['Bed'].isin([1,2])].copy()
        if combined.empty:
            logging.warning("No data for bed=1 or bed=2. Cannot plot combined eq.")
            return

        combined.sort_values('Last Rental Date', inplace=True)

        # Prepare daily aggregates
        combined['eq_volume'] = 1.0
        daily_agg = (
            combined.groupby('Last Rental Date', as_index=False)
            .agg({
                'eq_volume':'sum',
                'Last Rental Price': 'sum'
            })
        )
        daily_agg.rename(columns={'Last Rental Price':'sum_price'}, inplace=True)

        daily_agg.set_index('Last Rental Date', inplace=True)
        daily_agg.sort_index(inplace=True)

        # Rolling 3M
        daily_agg['Rolling_PV_3M'] = daily_agg['sum_price'].rolling('90D', min_periods=3).sum()
        daily_agg['Rolling_V_3M']  = daily_agg['eq_volume'].rolling('90D', min_periods=3).sum()
        daily_agg['VWAP_3M']       = (daily_agg['Rolling_PV_3M'] / daily_agg['Rolling_V_3M']).shift(1)

        # Rolling 12M
        daily_agg['Rolling_PV_12M'] = daily_agg['sum_price'].rolling('365D', min_periods=1).sum()
        daily_agg['Rolling_V_12M']  = daily_agg['eq_volume'].rolling('365D', min_periods=1).sum()
        daily_agg['VWAP_12M']       = (daily_agg['Rolling_PV_12M'] / daily_agg['Rolling_V_12M']).shift(1)

        # Average daily actual price
        daily_agg.reset_index(inplace=True)
        combined_daily_price = (
            combined.groupby('Last Rental Date', as_index=False)['Last Rental Price'].mean()
            .rename(columns={'Last Rental Price':'avg_price'})
        )
        merged = pd.merge(daily_agg, combined_daily_price, on='Last Rental Date', how='left')

        nobs = len(combined) 
        min_date = merged['Last Rental Date'].min().date()
        max_date = merged['Last Rental Date'].max().date()
        title_txt = (
            f"Combined EQ-Weighted VWAP for Bed=1 & Bed=2\n"
            f"Observations: {nobs}, Date Range: {min_date} => {max_date}"
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=merged['Last Rental Date'],
            y=merged['eq_volume'],
            marker=dict(color='lightgray'),
            opacity=0.2,
            showlegend=False,
            name='Combined Volume (1&2)'
        ), secondary_y=True)

        fig.add_trace(go.Scatter(
            x=merged['Last Rental Date'],
            y=merged['VWAP_3M'],
            mode='lines',
            name='Bed1+2 VWAP_3M_eq',
            line=dict(color='blue', width=2)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=merged['Last Rental Date'],
            y=merged['VWAP_12M'],
            mode='lines',
            name='Bed1+2 VWAP_12M_eq',
            line=dict(color='red', width=2)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=merged['Last Rental Date'],
            y=merged['avg_price'],
            mode='markers',
            name='Avg Actual Price (Daily, 1&2)',
            marker=dict(size=5, color='gray', opacity=0.5)
        ), secondary_y=False)

        fig.update_layout(
            title=title_txt,
            xaxis_title='Date',
            yaxis_title='Rental Price (Beds1+2 Combined EQ)',
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
        fig.update_yaxes(title_text="Combined Volume (1&2)", secondary_y=True, showgrid=False)
        # Move bars behind lines
        fig.data = fig.data[::-1]

        div_id = f"plot_div_1n2_{id(fig)}"
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id=div_id)

        # Updated snippet for shape events:
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

                    var pct = (y0!==0) ? ((y1 - y0)/y0)*100 : 0;
                    var x0d = new Date(x0).toLocaleDateString();
                    var x1d = new Date(x1).toLocaleDateString();

                    var newAnn = plot.layout.annotations ? plot.layout.annotations.slice() : [];
                    newAnn.push(
                        {{
                            x:x0, y:y0, xref:xref, yref:yref,
                            text:'Date: '+x0d+'<br>Price:'+y0.toFixed(2),
                            showarrow:true, arrowhead:7, ax:-40, ay:-40
                        }},
                        {{
                            x:x1, y:y1, xref:xref, yref:yref,
                            text:'Date: '+x1d+'<br>Price:'+y1.toFixed(2)+'<br>Change:'+pct.toFixed(2)+'%',
                            showarrow:true, arrowhead:7, ax:-40, ay:-40
                        }}
                    );
                    Plotly.relayout(plot, {{'annotations':newAnn}});
                }}
            }}
            if(eventdata['shapes[0]']===null){{
                Plotly.relayout(plot, {{'annotations':[]}});
            }}
        }});
        </script>
        """.strip()

        html_str = html_str.replace('</body>', custom_js + '</body>')

        os.makedirs(plot_dir, exist_ok=True)
        out_name = "eq_beds_1and2_combined.html"
        out_path = os.path.join(plot_dir, out_name)
        with open(out_path, 'w') as f:
            f.write(html_str)

        logging.info(f"Combined eq chart for bed=1 & 2 => {out_path}")

    except Exception as e:
        logging.error(f"Error in plot_eq_interactive_beds_1_and_2: {e}")
        logging.error(traceback.format_exc())
