# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import traceback
import os
from plotting import (
    plot_average_price_per_month,
    plot_cumulative_change
)
from utils import save_plotly_fig

def perform_eda(data, plot_dir):
    """
    Perform exploratory data analysis, including:
      - Plot median rental prices per suburb
      - Plot average price per month
      - Plot cumulative change
      - Additional correlation analysis
      - Interactive visualizations
    """
    try:
        logging.info("Starting EDA...")
        os.makedirs(plot_dir, exist_ok=True)

        plot_median_rental_price_per_suburb(data, plot_dir)

        date_col = 'Last Rental Date'
        price_col = 'Last Rental Price'
        if date_col in data.columns and price_col in data.columns:
            plot_average_price_per_month(data, date_col, price_col, plot_dir)
            plot_cumulative_change(data, date_col, price_col, plot_dir)
        else:
            logging.warning("Skipping monthly/cumulative plots due to missing columns.")

        advanced_statistical_tests(data, plot_dir)
        interactive_visualizations(data, plot_dir)

        logging.info("EDA completed successfully.")

    except Exception as e:
        logging.error(f"Error in perform_eda: {e}")
        logging.error(traceback.format_exc())
        raise

def plot_median_rental_price_per_suburb(data, plot_dir):
    """Plot median rental price by Suburb."""
    try:
        if 'Suburb' not in data.columns or 'Last Rental Price' not in data.columns:
            logging.warning("Cannot plot median rental price per suburb (missing columns).")
            return

        df_ = data.groupby('Suburb')['Last Rental Price'].median().reset_index()
        df_.rename(columns={'Last Rental Price':'Median Rental Price'}, inplace=True)
        df_.sort_values('Median Rental Price', ascending=False, inplace=True)

        plt.figure(figsize=(12,6))
        sns.barplot(data=df_, x='Suburb', y='Median Rental Price', palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title('Median Rental Price by Suburb')
        plt.tight_layout()

        fname = os.path.join(plot_dir, 'median_rental_price_by_suburb.png')
        plt.savefig(fname)
        plt.close()
        logging.info(f"Median rental price per suburb saved => {fname}")
    except Exception as e:
        logging.error(f"Error in plot_median_rental_price_per_suburb: {e}")
        raise

def advanced_statistical_tests(data, plot_dir):
    """Advanced correlation or ANOVA or other tests."""
    try:
        import numpy as np
        all_numeric_cols_in_df = data.select_dtypes(include=[np.number]).columns.tolist()

        # Define core non-calculated features and percentage differences
        core_features = [
            'Bath', 'Bed', 'Car', 'Days on Market', 'Last Rental Price',
            'Land Size (sqm)', 'Floor Size (sqm)', 'Year Built', 
            'Capital Value', 'Land Value', 'Postcode' # Postcode included if numeric
        ]

        # Define the specific VWAP features to include
        desired_vwaps = ['VWAP_3M_all_eq', 'VWAP_2Y_all_eq']

        # Combine the lists
        allowed_features = core_features + desired_vwaps

        # Select only allowed features that are actually numeric and present in the DataFrame
        cols_for_heatmap = [col for col in allowed_features if col in all_numeric_cols_in_df]
        
        # Explicitly remove 'Unnamed: 0' if it exists, as it's an index column
        if 'Unnamed: 0' in cols_for_heatmap:
            cols_for_heatmap.remove('Unnamed: 0')
            
        # Ensure no duplicates, though the construction method should prevent this
        cols_for_heatmap = sorted(list(set(cols_for_heatmap)))
        
        if not cols_for_heatmap:
            logging.warning("No columns selected for correlation heatmap after filtering. Skipping heatmap.")
            return

        logging.info(f"Columns selected for correlation heatmap: {cols_for_heatmap}")
        corr_ = data[cols_for_heatmap].corr()

        # Dynamically adjust figsize
        num_features = len(cols_for_heatmap)
        fig_width = max(12, num_features * 0.6) # Adjusted multiplier for width
        fig_height = max(10, num_features * 0.5) # Adjusted multiplier for height

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(corr_, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap (Selected Features)')
        plt.tight_layout()

        outpath = os.path.join(plot_dir, 'numeric_correlation_heatmap.png')
        plt.savefig(outpath)
        plt.close()
        logging.info(f"Correlation heatmap => {outpath}")

        # --------------------------------------------------
        # Interactive Global EQ VWAP plot (3M & 2Y) similar to static Matplotlib version
        # --------------------------------------------------
        try:
            required_cols_vwap = {'Last Rental Date', 'Last Rental Price', 'VWAP_3M_all_eq', 'VWAP_2Y_all_eq'}
            if required_cols_vwap.issubset(data.columns):
                vwap_df = data.sort_values('Last Rental Date')
                # Aggregate daily counts
                daily_cnt = (
                    vwap_df.groupby(vwap_df['Last Rental Date'].dt.date)['Last Rental Price']
                    .size()
                    .rename_axis('Date')
                    .to_frame('Daily_Count')
                )
                daily_cnt.index = pd.to_datetime(daily_cnt.index)

                fig_vwap = px.scatter(
                    vwap_df,
                    x='Last Rental Date',
                    y='Last Rental Price',
                    opacity=0.25,
                    size_max=4,
                    template='plotly_white',
                    labels={'Last Rental Price':'Price (NZD)', 'Last Rental Date':'Date'},
                    title='Global Equal-Weighted VWAP (2-Year) with Daily Means & Counts'
                )
                # Add 2Y VWAP line (thicker, distinct color)
                fig_vwap.add_scatter(
                    x=vwap_df['Last Rental Date'],
                    y=vwap_df['VWAP_2Y_all_eq'],
                    mode='lines',
                    name='2-Year VWAP (Eq All)',
                    line=dict(color='firebrick', width=3)
                )
                # Add total property count as legend (invisible marker)
                total_props = len(vwap_df)
                fig_vwap.add_scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(color='rgba(0,0,0,0)'),
                    name=f'Total Properties: {total_props}',
                    showlegend=True
                )
                # Add daily count bars on secondary axis
                fig_vwap.add_bar(x=daily_cnt.index, y=daily_cnt['Daily_Count'], name='Daily Count', marker_color='lightblue', opacity=0.3, yaxis='y2')

                fig_vwap.update_layout(
                    legend_orientation='h',
                    legend_title_text='',
                    hovermode='x unified',
                    yaxis=dict(title='Price (NZD)'),
                    yaxis2=dict(title='Daily Count', overlaying='y', side='right', showgrid=False)
                )

                out_vwap_html = os.path.join(plot_dir, 'interactive_global_eq_vwap.html')
                fig_vwap.write_html(out_vwap_html)
                logging.info(f"Interactive global EQ VWAP saved => {out_vwap_html}")
        except Exception as e:
            logging.error(f"Error generating interactive global VWAP: {e}")

    except Exception as e:
        logging.error(f"Error in advanced_statistical_tests: {e}")
        raise

def interactive_visualizations(data, plot_dir):
    """Example Plotly-based interactive visuals."""
    try:
        import plotly.express as px
        import numpy as np
        os.makedirs(plot_dir, exist_ok=True)

        # Example scatter: Floor Size vs. Price
        if 'Floor Size (sqm)' in data.columns and 'Last Rental Price' in data.columns:
            fig = px.scatter(
                data_frame=data,
                x='Floor Size (sqm)',
                y='Last Rental Price',
                color='Suburb',
                title='Floor Size vs. Rental Price by Suburb'
            )
            out_html = os.path.join(plot_dir, 'floor_vs_price_interactive.html')
            fig.write_html(out_html)
            logging.info(f"Interactive scatter saved => {out_html}")

        # --------------------------------------------------
        # Global VWAP-style rolling mean (3M / 2Y) using daily EQ logic
        # --------------------------------------------------
        try:
            if {'Last Rental Date', 'Last Rental Price'}.issubset(data.columns):
                tmp_glob = data[['Last Rental Date', 'Last Rental Price']].dropna().copy()
                tmp_glob['Last Rental Date'] = pd.to_datetime(tmp_glob['Last Rental Date'])

                # --------------------------------------------------
                # NEW OUTLIER FILTER: exclude prices > 9× overall mean
                # --------------------------------------------------
                global_mean_price = tmp_glob['Last Rental Price'].mean()
                price_threshold = global_mean_price * 9
                before_count = len(tmp_glob)
                tmp_glob = tmp_glob[tmp_glob['Last Rental Price'] <= price_threshold].copy()
                removed_obs = before_count - len(tmp_glob)
                if removed_obs:
                    logging.info(f"rolling_eq_mean_global: Removed {removed_obs} observations (price > 9× mean of {global_mean_price:.2f}).")

                # Daily equal-weighted mean (one observation per day)
                daily_agg = (
                    tmp_glob.groupby(tmp_glob['Last Rental Date'].dt.date)['Last Rental Price']
                    .agg(Daily_Mean_Price='mean', Daily_Count='size')
                    .rename_axis('Date')
                )
                daily_mean = daily_agg.copy()
                daily_mean.index = pd.to_datetime(daily_mean.index)

                # Rolling 90D and 730D means shifted by 1 day to mimic VWAP logic
                daily_mean['Rolling_Mean_1Y_eq'] = daily_mean['Daily_Mean_Price'].rolling(window='365D', min_periods=1).mean().shift(1)
                daily_mean['Rolling_Count_1Y'] = daily_mean['Daily_Count'].rolling(window='365D', min_periods=1).sum().shift(1)
                daily_mean.loc[daily_mean['Rolling_Count_1Y'] < 10, 'Rolling_Mean_1Y_eq'] = np.nan
                daily_mean.reset_index(inplace=True)

                fig_roll = px.line(
                    daily_mean,
                    x='Date',
                    y=['Rolling_Mean_1Y_eq'],
                    labels={'value':'Price (NZD)', 'Date':'Date', 'variable':'Metric'},
                    title='Global Equal-Weighted Rolling Mean (1Y) with Daily Means'
                )
                # Add daily mean scatter but keep it hidden by default to preserve clarity
                fig_roll.add_scatter(
                    x=daily_mean['Date'],
                    y=daily_mean['Daily_Mean_Price'],
                    mode='markers',
                    marker=dict(color='grey', size=3, opacity=0.15),
                    name='Daily Mean Price',
                    visible='legendonly'
                )
                # add total properties legend entry (invisible)
                total_props_rm = int(daily_mean['Daily_Count'].sum())
                fig_roll.add_scatter(x=[None], y=[None], mode='markers', marker=dict(color='rgba(0,0,0,0)'), name=f'Total Properties: {total_props_rm}', showlegend=True)
                # secondary axis count
                fig_roll.add_bar(x=daily_mean['Date'], y=daily_mean['Daily_Count'], name='Daily Count', marker_color='lightblue', opacity=0.3, yaxis='y2')
                fig_roll.update_layout(
                    height=700,
                    template='plotly_white',
                    hovermode='x unified',
                    margin=dict(l=60, r=60, t=80, b=60),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    yaxis=dict(title='Price (NZD)', tickformat=',', showgrid=True, zeroline=False),
                    yaxis2=dict(title='Daily Count', overlaying='y', side='right', showgrid=False, rangemode='tozero')
                )
                fig_roll.update_traces(selector=dict(mode='lines'), line_shape='spline')

                # Anchored VWAP (equal-weighted) starting from first data point
                try:
                    daily_mean['Cum_PV'] = (daily_mean['Daily_Mean_Price'] * daily_mean['Daily_Count']).cumsum()
                    daily_mean['Cum_V'] = daily_mean['Daily_Count'].cumsum()
                    daily_mean['Anchored_VWAP'] = np.where(
                        daily_mean['Cum_V'] >= 10,
                        daily_mean['Cum_PV'] / daily_mean['Cum_V'],
                        np.nan
                    )
                    fig_roll.add_scatter(
                        x=daily_mean['Date'],
                        y=daily_mean['Anchored_VWAP'],
                        mode='lines',
                        line=dict(color='green', width=2),
                        name='Anchored VWAP',
                    )
                except Exception as avwap_err:
                    logging.warning(f"Could not compute Anchored VWAP for rolling_eq_mean_global: {avwap_err}")

                # Adjust y-axis scaling using 2nd-98th percentiles to avoid extreme compression/expansion
                try:
                    q_low, q_high = daily_mean['Daily_Mean_Price'].quantile([0.02, 0.98])
                    fig_roll.update_yaxes(range=[q_low*0.95, q_high*1.05])
                except Exception as rng_err:
                    logging.warning(f"Could not set y-axis range for rolling price chart: {rng_err}")

                out_html_roll = os.path.join(plot_dir, 'rolling_eq_mean_global.html')
                try:
                    import plotly.io as pio
                    div_id = 'rolling_eq_mean_global'
                    post_script = """
var gd = document.getElementById('rolling_eq_mean_global');
if (!gd) { console.warn('Plot div not found: rolling_eq_mean_global'); } else {
  var anchorA = null, anchorB = null;
  function clearAnchors() {
    anchorA = null; anchorB = null;
    Plotly.relayout(gd, {shapes: [], annotations: []});
  }
  function formatCurrency(n) {
    try {
      return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'NZD', maximumFractionDigits: 0 }).format(n);
    } catch(e) {
      var sign = n >= 0 ? '' : '-';
      return sign + Math.round(Math.abs(n)).toString();
    }
  }
  function formatDate(v) {
    try {
      var d = new Date(v);
      if (!isNaN(d.getTime())) {
        return d.toISOString().slice(0,10);
      }
      return String(v);
    } catch(e) { return String(v); }
  }
  function drawDelta() {
    var dy = anchorB.y - anchorA.y;
    var pct = (dy / anchorA.y) * 100;
    var shapes = [
      {type:'line', x0:anchorA.x, y0:anchorA.y, x1:anchorB.x, y1:anchorB.y, line:{color:'rgba(30,144,255,0.85)', width:2}},
      {type:'line', x0:anchorA.x, x1:anchorA.x, y0:Math.min(anchorA.y,anchorB.y), y1:Math.max(anchorA.y,anchorB.y), line:{color:'rgba(30,144,255,0.4)', width:1, dash:'dot'}},
      {type:'line', x0:anchorB.x, x1:anchorB.x, y0:Math.min(anchorA.y,anchorB.y), y1:Math.max(anchorA.y,anchorB.y), line:{color:'rgba(220,20,60,0.4)', width:1, dash:'dot'}}
    ];
    var annotations = [
      {x: anchorA.x, y: anchorA.y, text:'A: ' + formatDate(anchorA.x) + ' | ' + formatCurrency(anchorA.y), ax:0, ay:-20, showarrow:true, arrowcolor:'#1e90ff', bgcolor:'white', bordercolor:'#1e90ff', borderpad:2},
      {x: anchorB.x, y: anchorB.y, text:'B: ' + formatDate(anchorB.x) + ' | ' + formatCurrency(anchorB.y), ax:0, ay:-20, showarrow:true, arrowcolor:'#dc143c', bgcolor:'white', bordercolor:'#dc143c', borderpad:2},
      {x: anchorB.x, y: (anchorA.y+anchorB.y)/2, ax: 0, ay: -40, xanchor:'left', showarrow:true,
        bgcolor:'rgba(255,255,255,0.9)', bordercolor:'#888', borderpad:6,
        text: 'Δ ' + formatCurrency(dy) + ' (' + pct.toFixed(2) + '%)'}
    ];
    Plotly.relayout(gd, {shapes: shapes, annotations: annotations});
  }
  gd.on('plotly_click', function(e) {
    if (!e || !e.points || !e.points.length) return;
    var p = e.points[0];
    var x = p.x, y = p.y;
    if (anchorA === null) {
      anchorA = {x:x, y:y};
      Plotly.relayout(gd, {annotations:[{x:x,y:y,text:'A: ' + formatDate(x) + ' | ' + formatCurrency(y), ax:0, ay:-20, showarrow:true, arrowcolor:'#1e90ff', bgcolor:'white', bordercolor:'#1e90ff', borderpad:2}], shapes:[] });
    } else if (anchorB === null) {
      anchorB = {x:x, y:y};
      drawDelta();
    } else {
      anchorA = {x:x, y:y};
      anchorB = null;
      Plotly.relayout(gd, {shapes:[], annotations:[{x:x,y:y,text:'A: ' + formatDate(x) + ' | ' + formatCurrency(y), ax:0, ay:-20, showarrow:true, arrowcolor:'#1e90ff', bgcolor:'white', bordercolor:'#1e90ff', borderpad:2}] });
    }
  });
  // Add a clear button overlay
  var btn = document.createElement('button');
  btn.textContent = 'Clear Anchors';
  btn.style.position = 'absolute';
  btn.style.top = '8px';
  btn.style.right = '8px';
  btn.style.zIndex = '1000';
  btn.style.padding = '6px 10px';
  btn.style.fontSize = '12px';
  btn.style.border = '1px solid #ccc';
  btn.style.background = '#fff';
  btn.style.cursor = 'pointer';
  btn.onclick = clearAnchors;
  // Ensure container is positioned
  var container = gd.parentNode;
  if (getComputedStyle(container).position === 'static') {
    container.style.position = 'relative';
  }
  container.appendChild(btn);
}
"""
                    html = pio.to_html(
                        fig_roll,
                        include_plotlyjs='cdn',
                        full_html=True,
                        div_id=div_id,
                        config={'displaylogo': False, 'modeBarButtonsToAdd': ['toggleSpikelines','hovercompare']},
                        post_script=post_script
                    )
                    with open(out_html_roll, 'w', encoding='utf-8') as f:
                        f.write(html)
                except Exception:
                    # Fallback to default write if injection fails
                    fig_roll.write_html(out_html_roll)
                logging.info(f"Global rolling EQ mean plot saved => {out_html_roll}")

        except Exception as e:
            logging.error(f"Error generating global rolling plot: {e}")

        # --------------------------------------------------
        # NEW: Global Price-per-sqm rolling mean (2Y) interactive
        # --------------------------------------------------
        try:
            ppsqm_cols = {'Last Rental Date', 'Last Rental Price', 'Floor Size (sqm)'}
            if ppsqm_cols.issubset(data.columns):
                ppsqm_df = data[list(ppsqm_cols)].dropna().copy()
                ppsqm_df = ppsqm_df[ppsqm_df['Floor Size (sqm)'] > 0]
                ppsqm_df['Last Rental Date'] = pd.to_datetime(ppsqm_df['Last Rental Date'])
                ppsqm_df['Price_per_sqm'] = ppsqm_df['Last Rental Price'] / ppsqm_df['Floor Size (sqm)']
                # remove inf or negative
                ppsqm_df = ppsqm_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price_per_sqm'])

                mean_ppsqm = ppsqm_df['Price_per_sqm'].mean()
                thresh_ppsqm = mean_ppsqm * 9
                before_pp = len(ppsqm_df)
                ppsqm_df = ppsqm_df[ppsqm_df['Price_per_sqm'] <= thresh_ppsqm].copy()
                removed_pp = before_pp - len(ppsqm_df)
                if removed_pp:
                    logging.info(f"rolling_eq_mean_global_ppsqm: Removed {removed_pp} obs (Price_per_sqm > 9× mean {mean_ppsqm:.2f}).")

                daily_pp = (
                    ppsqm_df.groupby(ppsqm_df['Last Rental Date'].dt.date)['Price_per_sqm']
                    .agg(Daily_Mean_PPSQM='mean', Daily_Count='size')
                    .rename_axis('Date')
                )
                daily_pp.index = pd.to_datetime(daily_pp.index)

                daily_pp['Rolling_Mean_1Y_ppsqm'] = daily_pp['Daily_Mean_PPSQM'].rolling(window='365D', min_periods=1).mean().shift(1)
                daily_pp['Rolling_Count_1Y'] = daily_pp['Daily_Count'].rolling(window='365D', min_periods=1).sum().shift(1)
                daily_pp.loc[daily_pp['Rolling_Count_1Y'] < 10, 'Rolling_Mean_1Y_ppsqm'] = np.nan
                daily_pp.reset_index(inplace=True)

                fig_pp = px.line(
                    daily_pp,
                    x='Date',
                    y=['Rolling_Mean_1Y_ppsqm'],
                    labels={'value':'Price per sqm (NZD/sqm)', 'Date':'Date', 'variable':'Metric'},
                    title='Global Equal-Weighted Rolling Mean (1Y) – Price per sqm'
                )

                # daily mean scatter hidden by default
                fig_pp.add_scatter(
                    x=daily_pp['Date'], y=daily_pp['Daily_Mean_PPSQM'], mode='markers',
                    marker=dict(color='grey', size=3, opacity=0.15), name='Daily Mean PPSQM', visible='legendonly'
                )

                total_props_pp = int(daily_pp['Daily_Count'].sum())
                fig_pp.add_scatter(
                    x=[None], y=[None], mode='markers', marker=dict(color='rgba(0,0,0,0)'),
                    name=f'Total Properties: {total_props_pp}', showlegend=True
                )

                fig_pp.add_bar(x=daily_pp['Date'], y=daily_pp['Daily_Count'], name='Daily Count',
                               marker_color='lightblue', opacity=0.3, yaxis='y2')
                fig_pp.update_layout(
                    height=700,
                    template='plotly_white',
                    hovermode='x unified',
                    margin=dict(l=60, r=60, t=80, b=60),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    yaxis=dict(title='Price per sqm (NZD/sqm)', tickformat=',', showgrid=True, zeroline=False),
                    yaxis2=dict(title='Daily Count', overlaying='y', side='right', showgrid=False, rangemode='tozero')
                )
                fig_pp.update_traces(selector=dict(mode='lines'), line_shape='spline')

                # Anchored VWAP for price per sqm (equal-weighted)
                try:
                    daily_pp['Cum_PV'] = (daily_pp['Daily_Mean_PPSQM'] * daily_pp['Daily_Count']).cumsum()
                    daily_pp['Cum_V'] = daily_pp['Daily_Count'].cumsum()
                    daily_pp['Anchored_VWAP_PPSQM'] = np.where(
                        daily_pp['Cum_V'] >= 10,
                        daily_pp['Cum_PV'] / daily_pp['Cum_V'],
                        np.nan
                    )
                    fig_pp.add_scatter(
                        x=daily_pp['Date'],
                        y=daily_pp['Anchored_VWAP_PPSQM'],
                        mode='lines',
                        line=dict(color='green', width=2),
                        name='Anchored VWAP PPSQM',
                    )
                except Exception as avwap_pp_err:
                    logging.warning(f"Could not compute Anchored VWAP PPSQM: {avwap_pp_err}")

                # Adjust y-axis scaling using 2nd-98th percentiles to avoid extreme compression/expansion
                try:
                    q_low_pp, q_high_pp = daily_pp['Daily_Mean_PPSQM'].quantile([0.02, 0.98])
                    fig_pp.update_yaxes(range=[q_low_pp*0.95, q_high_pp*1.05])
                except Exception as rng_pp_err:
                    logging.warning(f"Could not set y-axis range for PPSQM chart: {rng_pp_err}")

                out_html_pp = os.path.join(plot_dir, 'rolling_eq_mean_global_ppsqm.html')
                fig_pp.write_html(out_html_pp)
                logging.info(f"Global rolling EQ mean (price per sqm) plot saved => {out_html_pp}")
        except Exception as e:
            logging.error(f"Error generating global rolling price-per-sqm plot: {e}")

        # New: Price per sqm by suburb interactive bar plot
        req_cols = {'Last Rental Price', 'Floor Size (sqm)', 'Suburb'}
        if req_cols.issubset(data.columns):
            tmp = data.copy()
            tmp = tmp[tmp['Floor Size (sqm)'] > 0]
            tmp['Price_per_sqm'] = tmp['Last Rental Price'] / tmp['Floor Size (sqm)']

            # --------------------------
            #  Outlier removal (suburb-level)
            #  Any observation > 5× median price_per_sqm of its suburb is dropped.
            # --------------------------
            suburb_median_ppsqm = tmp.groupby('Suburb')['Price_per_sqm'].median()
            tmp['suburb_median'] = tmp['Suburb'].map(suburb_median_ppsqm)
            mask_outlier = tmp['Price_per_sqm'] > tmp['suburb_median'] * 5
            num_outliers = mask_outlier.sum()
            if num_outliers:
                logging.info(f"Filtering out {num_outliers} extreme price_per_sqm points (>5× median of suburb).")
            tmp = tmp[~mask_outlier].copy()

            # Compute sample size for each suburb AFTER outlier removal
            counts = tmp.groupby('Suburb', as_index=False).size().rename(columns={'size': 'SampleSize'})

            # Global median for thresholding
            global_med_all = tmp['Price_per_sqm'].median()

            # Apply sample-size (>3) AND median >= 0.5 * global median filters
            suburb_medians_after = tmp.groupby('Suburb')['Price_per_sqm'].median()
            qual_suburbs = suburb_medians_after[suburb_medians_after >= global_med_all * 0.5].index

            valid_suburbs_ser = counts[(counts['SampleSize'] > 3) & (counts['Suburb'].isin(qual_suburbs))]['Suburb']
            valid_suburbs = valid_suburbs_ser.tolist()

            tmp = tmp[tmp['Suburb'].isin(valid_suburbs)].copy()
            counts = counts[counts['Suburb'].isin(valid_suburbs)]

            # Merge counts back for hover info
            tmp = tmp.merge(counts, on='Suburb', how='left')

            # Sort suburbs by median for tick order
            med_order = (
                tmp.groupby('Suburb')['Price_per_sqm']
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )

            new_ticktext = [f"{sub}\n(n={counts.loc[counts.Suburb==sub, 'SampleSize'].iloc[0]})" for sub in med_order]
            fig_box = px.box(
                tmp,
                x='Suburb',
                y='Price_per_sqm',
                points='all',
                hover_data={'SampleSize': True, 'Price_per_sqm': ':.2f'},
                title='Distribution of Annual Rental Price per sqm by Suburb\n(arranged by median)',
            )

            fig_box.update_layout(
                xaxis={'title': 'Suburb (sample size in label)', 'tickangle': -45, 'tickmode': 'array', 'tickvals': med_order, 'ticktext': new_ticktext},
                yaxis_title='Price per sqm (annual)',
            )

            out_html_pp = os.path.join(plot_dir, 'price_per_sqm_by_suburb.html')
            fig_box.write_html(out_html_pp)
            logging.info(f"Price-per-sqm distribution plot saved => {out_html_pp}")

            # --------------------------------------------------
            # Median (bar height) + Average (label) chart
            # --------------------------------------------------
            agg_stats = (
                tmp.groupby('Suburb')
                .agg(MedianPrice=('Price_per_sqm', 'median'), AvgPrice=('Price_per_sqm', 'mean'), SampleSize=('Price_per_sqm', 'size'))
                .reset_index()
                .sort_values('MedianPrice', ascending=False)
            )

            # Global stats for reference lines
            global_median_ppsqm = tmp['Price_per_sqm'].median()
            global_avg_ppsqm = tmp['Price_per_sqm'].mean()

            fig_med = px.bar(
                agg_stats,
                x='Suburb',
                y='MedianPrice',
                color='MedianPrice',
                color_continuous_scale='Viridis',
                text=agg_stats['MedianPrice'].round(0),
                hover_data={'AvgPrice':':.2f','MedianPrice':':.2f','SampleSize':True},
                title='Median (inside bar) vs Mean (hover) of Annual Rental Price per sqm by Suburb',
            )
            fig_med.update_coloraxes(colorbar_title='Median NZD/sqm')
            fig_med.update_traces(textposition='inside', texttemplate='%{text:.0f}')
            fig_med.update_layout(
                xaxis_title='Suburb',
                yaxis_title='Median Annual Rental Price per sqm (NZD/sqm)',
                xaxis_tickangle=-45,
            )
            fig_med.update_yaxes(tickformat=',')

            # Add reference lines
            fig_med.add_hline(y=global_median_ppsqm, line_dash="dash", line_color="green",
                              annotation_text=f"Global Median {global_median_ppsqm:.0f}", annotation_position="bottom right")
            fig_med.add_hline(y=global_avg_ppsqm, line_dash="dot", line_color="red",
                              annotation_text=f"Global Avg {global_avg_ppsqm:.0f}", annotation_position="top right")

            out_html_med = os.path.join(plot_dir, 'median_avg_price_per_sqm_by_suburb.html')
            fig_med.write_html(out_html_med)
            logging.info(f"Median vs average price-per-sqm plot saved => {out_html_med}")

            # --------------------------------------------------
            # Interactive heatmap of median price per sqm by suburb
            # --------------------------------------------------
            try:
                # Ensure Suburb_LOO numeric feature exists for correlation
                if 'Suburb_LOO' not in tmp.columns and 'Suburb' in tmp.columns:
                    try:
                        from category_encoders import LeaveOneOutEncoder as _LOO
                        loo_enc = _LOO(cols=['Suburb'], sigma=0.1, random_state=42)
                        tmp['Suburb_LOO'] = loo_enc.fit_transform(tmp[['Suburb']], tmp['Last Rental Price'])['Suburb']
                    except Exception as loo_err:
                        logging.warning(f"Could not compute Suburb_LOO for heatmap: {loo_err}")

                heat_df = agg_stats.set_index('Suburb')[['MedianPrice','AvgPrice']].T  # 2 metrics rows
                fig_heat = px.imshow(
                    heat_df,
                    color_continuous_scale='Viridis',
                    labels=dict(x='Suburb', y='Metric', color='NZD/sqm'),
                    aspect='auto',
                    title='Heatmap of Median and Mean Annual Rental Price per sqm by Suburb',
                    text_auto='.0f'
                )
                # Format axes for readability
                fig_heat.update_xaxes(side='bottom', tickangle=-45)
                fig_heat.update_yaxes(autorange='reversed')
                fig_heat.update_traces(hovertemplate='Value: %{z:.2f} NZD/sqm<extra></extra>')

                out_html_heat = os.path.join(plot_dir, 'heatmap_median_price_per_sqm_by_suburb.html')
                fig_heat.write_html(out_html_heat)
                logging.info(f"Heatmap of median price-per-sqm saved => {out_html_heat}")

                # --------------------------------------------------
                # Interactive correlation heatmap for user-specified features
                # --------------------------------------------------
                corr_features = [
                    'Bath', 'Capital Value', 'Car', 'Days on Market',
                    'Floor Size (sqm)', 'Land Size (sqm)', 'Land Value',
                    'Last Rental Price', 'Postcode', 'Year Built', 'Price_per_sqm', 'Suburb_LOO'
                ]
                # Recompute price_per_sqm to ensure consistency
                tmp['Price_per_sqm'] = tmp['Last Rental Price'] / tmp['Floor Size (sqm)']
                # Only keep features that are actually present to avoid KeyError (e.g., 'Land Value' may be absent)
                available = [c for c in corr_features if c in tmp.columns]
                if not available:
                    logging.warning("Correlation heatmap skipped: requested features not present in dataset.")
                    return
                numeric_subset = tmp[available].dropna(axis=1, how='all')
                if not numeric_subset.empty:
                    corr_matrix = numeric_subset.corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        labels=dict(x='Feature', y='Feature', color='Correlation'),
                        title='Interactive Correlation Heatmap (Selected Features)'
                    )
                    fig_corr.update_xaxes(side='bottom', tickangle=-45)
                    out_corr_html = os.path.join(plot_dir, 'interactive_correlation_heatmap.html')
                    fig_corr.write_html(out_corr_html)
                    logging.info(f"Interactive correlation heatmap saved => {out_corr_html}")
                else:
                    logging.warning("Correlation heatmap skipped: none of the specified features available after filtering.")
            except Exception as e:
                logging.error(f"Error generating heatmap: {e}")
        else:
            missing = req_cols - set(data.columns)
            logging.warning(f"Missing columns {missing}; skipping price-per-sqm interactive plot.")

        # --------------------------------------------------
        # Gross Rental Yield per Suburb – Median vs Mean (interactive)
        # --------------------------------------------------
        yield_cols = {'Last Rental Price', 'Capital Value', 'Suburb'}
        if yield_cols.issubset(data.columns):
            ytmp = data.copy()
            ytmp = ytmp[ytmp['Capital Value'] > 0]
            ytmp['Gross_Yield'] = ytmp['Last Rental Price'] / ytmp['Capital Value'] * 100

            # Remove extreme yields (> 5× median of suburb)
            suburb_med_yield = ytmp.groupby('Suburb')['Gross_Yield'].median()
            ytmp['suburb_med'] = ytmp['Suburb'].map(suburb_med_yield)
            ytmp = ytmp[ytmp['Gross_Yield'] <= ytmp['suburb_med'] * 5].copy()

            # Sample-size filter (>3) and yield ≥ 0.5 × global median
            counts_y = ytmp.groupby('Suburb').size()
            global_med_yield = ytmp['Gross_Yield'].median()
            valid_subs = counts_y[counts_y > 3].index
            valid_subs = [s for s in valid_subs if suburb_med_yield[s] >= global_med_yield * 0.5]
            ytmp = ytmp[ytmp['Suburb'].isin(valid_subs)].copy()

            if not ytmp.empty:
                agg_yield = (
                    ytmp.groupby('Suburb')
                    .agg(MedianYield=('Gross_Yield', 'median'), AvgYield=('Gross_Yield', 'mean'), SampleSize=('Gross_Yield', 'size'))
                    .reset_index()
                    .sort_values('MedianYield', ascending=False)
                )

                fig_yield = px.bar(
                    agg_yield,
                    x='Suburb',
                    y='MedianYield',
                    color='MedianYield',
                    color_continuous_scale='Viridis',
                    text=agg_yield['MedianYield'].round(2),
                    hover_data={'AvgYield':':.2f','MedianYield':':.2f','SampleSize':True},
                    title='Gross Rental Yield by Suburb — Median vs Mean',
                    template='plotly_white',
                    height=600
                )
                fig_yield.update_coloraxes(colorbar_title='Median Yield %')
                fig_yield.update_traces(textposition='inside', texttemplate='%{text:.2f}%')
                fig_yield.update_layout(
                    xaxis_title='Suburb',
                    yaxis_title='Gross Rental Yield (%)',
                    xaxis_tickangle=-45
                )
                fig_yield.add_hline(y=global_med_yield, line_dash='dash', line_color='green', annotation_text=f"Global Median {global_med_yield:.2f}%", annotation_position='bottom right')
                global_avg_yield = ytmp['Gross_Yield'].mean()
                fig_yield.add_hline(y=global_avg_yield, line_dash='dot', line_color='red', annotation_text=f"Global Avg {global_avg_yield:.2f}%", annotation_position='top right')

                save_plotly_fig(fig_yield, 'median_avg_yield_by_suburb', plot_dir)
                logging.info('Gross rental yield plot saved => median_avg_yield_by_suburb.html')
        else:
            logging.warning(f"Missing columns {yield_cols - set(data.columns)}; skipping rental-yield plot.")

    except Exception as e:
        logging.error(f"Error in interactive_visualizations: {e}")
        raise
