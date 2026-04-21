import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats as sp_stats
import climate_core
import subprocess, sys

try:
    import statsmodels.api as sm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels", "-q"])
    import statsmodels.api as sm

# Auto-install openai if not present (Python 3.12 compatible)
try:
    import openai
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "-q"])
    import openai

@st.cache_data
def cached_load_data(file_obj):
    return climate_core.load_data(file_obj)

@st.cache_data
def cached_calculate_all_indices(df_clean, hemisphere, thresh_fd, thresh_su, thresh_id, thresh_tr, thresh_rnn, resolution):
    return climate_core.calculate_all_indices(
        df_clean, hemisphere=hemisphere,
        thresh_fd=thresh_fd, thresh_su=thresh_su, thresh_id=thresh_id, thresh_tr=thresh_tr,
        thresh_rnn=thresh_rnn, resolution=resolution
    )

st.set_page_config(page_title="Climatrend", layout="wide", page_icon="🌍")

st.markdown("""
<style>
    .reportview-container { background-color: #f0f2f6; }
    h1 { color: #1e3a8a; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #e0e0e0;
        padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Climatrend: Climate Analysis Suite")
st.markdown("Advanced ETCCDI indices calculator with interactive visualization and comprehensive statistical analytics.")

INDEX_METADATA = {
    'FD': {'definition': 'Frost Days: Days when temperature goes below threshold (default 0°C) at night', 'unit': 'Days', 'significance': 'Shows cold risk for crops and people'},
    'SU': {'definition': 'Summer Days: Days when daytime temperature exceeds threshold (default 25°C)', 'unit': 'Days', 'significance': 'Indicates warm/hot conditions'},
    'ID': {'definition': 'Ice Days: Days when temperature stays below threshold (default 0°C) even in daytime', 'unit': 'Days', 'significance': 'Measures extreme cold conditions'},
    'TR': {'definition': 'Tropical Nights: Nights when temperature stays above threshold (default 20°C)', 'unit': 'Days', 'significance': 'Shows night-time heat discomfort'},
    'FD0': {'definition': 'Frost Days (Standard): Days when TMIN < 0°C', 'unit': 'Days', 'significance': 'Shows cold risk for crops and people'},
    'SU25': {'definition': 'Summer Days (Standard): Days when TMAX > 25°C', 'unit': 'Days', 'significance': 'Indicates warm/hot conditions'},
    'ID0': {'definition': 'Ice Days (Standard): Days when TMAX < 0°C', 'unit': 'Days', 'significance': 'Measures extreme cold conditions'},
    'TR20': {'definition': 'Tropical Nights (Standard): Nights when TMIN > 20°C', 'unit': 'Days', 'significance': 'Shows night-time heat discomfort'},
    'FD_user': {'definition': 'Frost Days (Custom): Days when TMIN < user threshold', 'unit': 'Days', 'significance': 'Customizable cold tracking'},
    'SU_user': {'definition': 'Summer Days (Custom): Days when TMAX > user threshold', 'unit': 'Days', 'significance': 'Customizable heat tracking'},
    'ID_user': {'definition': 'Ice Days (Custom): Days when TMAX < user threshold', 'unit': 'Days', 'significance': 'Customizable severe cold tracking'},
    'TR_user': {'definition': 'Tropical Nights (Custom): Nights when TMIN > user threshold', 'unit': 'Days', 'significance': 'Customizable night heat tracking'},
    'GSL': {'definition': 'Growing Season Length: Length of the warm period suitable for plant growth', 'unit': 'Days', 'significance': 'Important for agriculture planning'},
    'TXx': {'definition': 'Max Tmax: Hottest day of the year', 'unit': '°C', 'significance': 'Measures peak heat intensity'},
    'TNn': {'definition': 'Min Tmin: Coldest night of the year', 'unit': '°C', 'significance': 'Identifies extreme cold events'},
    'TXn': {'definition': 'Min Tmax: Coldest daytime temperature', 'unit': '°C', 'significance': 'Shows severity of cold days'},
    'TNx': {'definition': 'Max Tmin: Warmest night of the year', 'unit': '°C', 'significance': 'Indicates heat stress at night'},
    'TN10p': {'definition': 'Cool Nights: Percentage of unusually cold nights', 'unit': '%', 'significance': 'Tracks cooling trends'},
    'TX10p': {'definition': 'Cool Days: Percentage of unusually cool days', 'unit': '%', 'significance': 'Measures cold variability'},
    'TN90p': {'definition': 'Warm Nights: Percentage of unusually warm nights', 'unit': '%', 'significance': 'Important for urban heat studies'},
    'TX90p': {'definition': 'Warm Days: Percentage of unusually hot days', 'unit': '%', 'significance': 'Tracks frequency of heatwaves'},
    'WSDI': {'definition': 'Warm Spell Duration: Days in periods of 6+ consecutive days where Tmax > 90th percentile', 'unit': 'Days', 'significance': 'Shows length of sustained heat waves'},
    'CSDI': {'definition': 'Cold Spell Duration: Days in periods of 6+ consecutive days where Tmin < 10th percentile', 'unit': 'Days', 'significance': 'Shows length of sustained cold waves'},
    'HWDI': {'definition': 'Heatwave Duration Index: Number of days in 3+ consecutive day spells where TMAX > 95th percentile', 'unit': 'Days', 'significance': 'Measures extreme heat stress duration'},
    'HWI': {'definition': 'Heatwave Intensity: Cumulative excess temperature during heatwaves', 'unit': '°C', 'significance': 'Measures the severity and physical stress of heatwaves'},
    'DTR': {'definition': 'Diurnal Temperature Range: Mean difference between Tmax and Tmin', 'unit': '°C', 'significance': 'Shows changes in day/night temperature spread'},
    'RX1day': {'definition': 'Max 1-day Rainfall: Highest rainfall in a single day', 'unit': 'mm', 'significance': 'Indicates flash flood risk'},
    'RX5day': {'definition': 'Max 5-day Rainfall: Highest rainfall over 5 consecutive days', 'unit': 'mm', 'significance': 'Indicates flood and landslide risk'},
    'SDII': {'definition': 'Rainfall Intensity: Average rainfall on rainy days', 'unit': 'mm/day', 'significance': 'Shows how intense rain events are'},
    'R10mm': {'definition': 'Heavy Rain Days: Number of days with rainfall >= 10 mm', 'unit': 'Days', 'significance': 'Tracks moderate heavy rain'},
    'R20mm': {'definition': 'Very Heavy Rain Days: Number of days with rainfall >= 20 mm', 'unit': 'Days', 'significance': 'Tracks severe rainfall events'},
    'R95p': {'definition': 'Very Wet Days: Total rainfall from very heavy rain events (top 5%)', 'unit': 'mm', 'significance': 'Measures contribution of extreme rain'},
    'R99p': {'definition': 'Extremely Wet Days: Total rainfall from extreme events (top 1%)', 'unit': 'mm', 'significance': 'Identifies rare extreme rainfall'},
    'PRCPTOT': {'definition': 'Total Rainfall: Total rainfall in wet days over a year', 'unit': 'mm', 'significance': 'Shows overall water availability'},
    'CDD': {'definition': 'Consecutive Dry Days: Longest period without rain', 'unit': 'Days', 'significance': 'Indicates drought conditions'},
    'CWD': {'definition': 'Consecutive Wet Days: Longest period with continuous rain', 'unit': 'Days', 'significance': 'Indicates prolonged wet/flood conditions'},
    'R1mm': {'definition': 'Wet Days: Number of days with at least 1 mm rainfall', 'unit': 'Days', 'significance': 'Shows rainfall frequency'},
    'Rnn': {'definition': 'User-defined precipitation days: Number of days with rainfall >= user threshold', 'unit': 'Days', 'significance': 'Customizable extreme rainfall tracking'},
    'MAM_TMAX_Ave': {'definition': 'Spring TMAX', 'unit': '°C', 'significance': 'March-April-May Average Maximum Temperature.'},
    'DJF_TMAX_Ave': {'definition': 'Winter TMAX', 'unit': '°C', 'significance': 'December-January-February Average Maximum Temperature.'},
    'JJAS_PRCP_Ave': {'definition': 'Monsoon PRCP', 'unit': 'mm/day', 'significance': 'June-July-August-September Average Precipitation.'}
}

def create_plotly_figure(df, col_name, thresholds, resolution, pub_mode=False):
    meta = INDEX_METADATA.get(col_name, {'definition': 'Index data', 'unit': '', 'significance': ''})
    thresh_val = thresholds.get(col_name)
    title_suffix = f" (Threshold: {thresh_val})" if thresh_val is not None else ""
    
    x_axis_title = "Year" if resolution == "Annual" else "Year-Month"
    
    fig = go.Figure()
    valid_data = df[col_name].dropna()
    if valid_data.empty: return fig
    
    x_hover_label = "Year" if resolution == "Annual" else "Date"
    hover_text = (
        f"<b>{col_name}</b><br>"
        f"{x_hover_label}: %{{x}}<br>"
        f"Value: %{{y}} {meta['unit']}<br>"
        f"<i>{meta['definition']}</i><br>"
        f"<span style='font-size:0.9em; color:#666;'>{meta.get('significance', '')}</span>"
        "<extra></extra>"
    )
    
    if pub_mode:
        fig.add_trace(go.Scatter(
            x=valid_data.index, y=valid_data.values,
            mode='lines+markers', name='Observed',
            line=dict(color='gray', width=1),
            marker=dict(symbol='circle-open', color='black', size=7, line=dict(width=1)),
            hovertemplate=hover_text
        ))
    else:
        fig.add_trace(go.Scatter(
            x=valid_data.index, y=valid_data.values,
            mode='lines+markers', name='Observed',
            line=dict(color='#1e40af', width=2.5),
            marker=dict(size=6, color='#1e40af'),
            hovertemplate=hover_text
        ))
    
    change_point = compute_smk_change_point(valid_data)
    stats_html = ""
    pub_stats_text = ""
    
    if change_point is not None and change_point in valid_data.index:
        idx_cp = valid_data.index.get_loc(change_point)
        seg1 = valid_data.iloc[:idx_cp+1]
        seg2 = valid_data.iloc[idx_cp:]
        
        slope1, se1, r2_1, pval1 = compute_trend_stats(seg1)
        slope2, se2, r2_2, pval2 = compute_trend_stats(seg2)
        
        if slope1 is not None and slope2 is not None:
            sig1 = "" if pval1 >= 0.05 else ("*" if pval1 >= 0.01 else ("**" if pval1 >= 0.001 else "***"))
            sig2 = "" if pval2 >= 0.05 else ("*" if pval2 >= 0.01 else ("**" if pval2 >= 0.001 else "***"))
            
            stats_html = f"<br><span style='font-size:13px; color:gray; font-weight:normal;'>Pre-Shift: {slope1:.3f}/yr (P={pval1:.3f}{sig1}) &nbsp;|&nbsp; Post-Shift: {slope2:.3f}/yr (P={pval2:.3f}{sig2})</span>"
            pub_stats_text = f"Pre-Shift: Slope= {slope1:.3f} (p= {pval1:.3f})   |   Post-Shift: Slope= {slope2:.3f} (p= {pval2:.3f})"
            
            x_vals_1 = np.arange(len(seg1))
            trend_y1 = slope1 * x_vals_1 + (seg1.mean() - slope1 * x_vals_1.mean())
            fig.add_trace(go.Scatter(x=seg1.index, y=trend_y1, mode='lines', name='Pre-Shift Trend', line=dict(color='#ea580c', width=2, dash='dash' if not pub_mode else 'solid')))
            
            x_vals_2 = np.arange(len(seg2))
            trend_y2 = slope2 * x_vals_2 + (seg2.mean() - slope2 * x_vals_2.mean())
            fig.add_trace(go.Scatter(x=seg2.index, y=trend_y2, mode='lines', name='Post-Shift Trend', line=dict(color='#b91c1c', width=2, dash='dash' if not pub_mode else 'solid')))
            
        fig.add_vline(x=change_point, line_width=2, line_dash="dash", line_color="gray", annotation_text="Shift Point")
    else:
        slope_stat, slope_se, r2, pval = compute_trend_stats(valid_data)
        if slope_stat is not None:
            sig_star = "" if pval >= 0.05 else ("*" if pval >= 0.01 else ("**" if pval >= 0.001 else "***"))
            stats_html = f"<br><span style='font-size:13px; color:gray; font-weight:normal;'>R² = {r2:.4f} | P-Value = {pval:.4f}{sig_star} | Slope = {slope_stat:.4f}/yr</span>"
            pub_stats_text = f"R2= {r2*100:.1f} p-value= {pval:.3f} Slope estimate= {slope_stat:.3f} Slope error= {slope_se:.3f}"
            
            x_vals = np.arange(len(valid_data))
            trend_y = slope_stat * x_vals + (valid_data.mean() - slope_stat * x_vals.mean())
            fig.add_trace(go.Scatter(
                x=valid_data.index, y=trend_y,
                mode='lines', name='Linear Trend',
                line=dict(color='black' if pub_mode else '#dc2626', width=2, dash='solid' if pub_mode else 'dash')
            ))
            
    if pub_mode and len(valid_data) > 10:
        lowess = sm.nonparametric.lowess(valid_data.values, np.arange(len(valid_data)), frac=0.3)
        fig.add_trace(go.Scatter(
            x=valid_data.index, y=lowess[:, 1],
            mode='lines', name='LOESS',
            line=dict(color='black', width=2, dash='dash')
        ))
    
    if pub_mode:
        fig.update_layout(
            title=dict(text=f"<b>{col_name}</b>", x=0.5, y=0.95, font=dict(family='Arial', size=20, color='black')),
            paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
            margin=dict(l=70, r=40, t=60, b=90),
            font=dict(family='Arial', size=14, color='black'),
            xaxis=dict(
                title=dict(text=f"{x_axis_title}", font=dict(size=16, color='black', family='Arial')),
                tickfont=dict(size=14, color='black', family='Arial'),
                showline=True, linewidth=1.5, linecolor='black',
                mirror='allticks', ticks='inside', ticklen=6, tickwidth=1.5,
                showgrid=False, zeroline=False
            ),
            yaxis=dict(
                title=dict(text=f"{col_name}", font=dict(size=16, color='black', family='Arial')),
                tickfont=dict(size=14, color='black', family='Arial'),
                showline=True, linewidth=1.5, linecolor='black',
                mirror='allticks', ticks='inside', ticklen=6, tickwidth=1.5,
                showgrid=False, zeroline=False
            ),
            showlegend=False
        )
        if pub_stats_text:
            fig.add_annotation(
                x=0.5, y=-0.22, xref="paper", yref="paper",
                text=pub_stats_text, showarrow=False,
                font=dict(family='Arial', size=14, color='black'),
                align="center", xanchor="center", yanchor="top"
            )
    else:
        fig.update_layout(
            title=f"<b>{col_name}</b>{title_suffix}{stats_html}",
            xaxis_title=x_axis_title,
            yaxis_title=f"Value ({meta['unit']})" if meta['unit'] else "Value",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=80, b=40),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)')
        )

    return fig

def compute_trend_stats(series: pd.Series):
    """Return (slope, slope_stderr, r_squared, p_value) for a 1-D numeric series."""
    valid = series.dropna()
    if len(valid) < 3:
        return None, None, None, None
    x = np.arange(len(valid), dtype=float)
    result = sp_stats.linregress(x, valid.values.astype(float))
    r2 = result.rvalue ** 2
    return result.slope, result.stderr, r2, result.pvalue

def compute_smk_change_point(series: pd.Series):
    """Calculates Sequential Mann-Kendall to find change point."""
    valid = series.dropna()
    n = len(valid)
    if n < 10: return None
    
    x = valid.values
    def calc_u(data):
        t = np.zeros(len(data))
        for i in range(1, len(data)):
            s = np.sum(np.sign(data[i] - data[:i]))
            t[i] = t[i-1] + s
        E_t = np.array([i*(i-1)/4 for i in range(1, len(data)+1)])
        Var_t = np.array([i*(i-1)*(2*i+5)/72 for i in range(1, len(data)+1)])
        u = np.zeros(len(data))
        with np.errstate(divide='ignore', invalid='ignore'):
            u[1:] = (t[1:] - E_t[1:]) / np.sqrt(Var_t[1:])
        return np.nan_to_num(u)
        
    u_fwd = calc_u(x)
    u_bwd = calc_u(x[::-1])[::-1]
    
    diff = u_fwd - u_bwd
    crossings = np.where(np.diff(np.sign(diff)))[0]
    
    valid_crossings = [c for c in crossings if 5 < c < n-5]
    if valid_crossings:
        best_c = max(valid_crossings, key=lambda c: abs(u_fwd[c]))
        if abs(u_fwd[best_c]) > 1.96: # 95% confidence
            return valid.index[best_c]
    return None

st.sidebar.header("⚙️ Configuration Panel")
uploaded_file = st.sidebar.file_uploader("Upload Climate Data (.csv or .txt)", type=["csv", "txt"])

st.sidebar.markdown("### Analysis Settings")
resolution = st.sidebar.radio("Analysis Resolution", ["Annual", "Monthly"], index=0)
publication_mode = st.sidebar.checkbox("🎓 Publication Mode (Journal Ready)", value=False, help="Removes titles, changes to high-contrast B&W, and enables SVG high-res export.")

st.sidebar.markdown("### Location Settings")
hemisphere = st.sidebar.radio("Hemisphere (for GSL)", ["Northern", "Southern"], index=0)

st.sidebar.markdown("### Temperature Thresholds")
thresh_su = st.sidebar.number_input("Upper threshold of Daily Max Temp (SU) > °C", value=25.0)
thresh_id = st.sidebar.number_input("Lower threshold of Daily Max Temp (ID) < °C", value=0.0)
thresh_tr = st.sidebar.number_input("Upper threshold of Daily Min Temp (TR) > °C", value=20.0)
thresh_fd = st.sidebar.number_input("Lower threshold of Daily Min Temp (FD) < °C", value=0.0)

st.sidebar.markdown("### Rainfall Thresholds")
thresh_rnn = st.sidebar.number_input("User-defined Rain (Rnn) >= mm", value=30.0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

if uploaded_file is not None:
    try:
        df = cached_load_data(uploaded_file)
        df_all_stations = df.copy()
        
        if 'STATION' in df.columns:
            st.sidebar.markdown("### Station Selection")
            stations = df['STATION'].unique()
            selected_station = st.sidebar.selectbox("🎯 Select Station", stations)
            df = df[df['STATION'] == selected_station].reset_index(drop=True)
            
        main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6, main_tab7 = st.tabs([
            "🔍 Data Preview", "🛡️ QC Report", "📊 Calculated Indices",
            "📈 Professional Analytics Hub", "🌊 Extreme Probability", "🤖 AI Research Assistant", "🗺️ Station Map"
        ])
        
        with main_tab1:
            st.markdown("### 📅 Advanced Seasonal & Month Filtering")
            st.caption("Select specific months to isolate your ETCCDI calculations (e.g., only calculate summer heatwaves). Note: Selecting non-consecutive months will cause spell-duration indices (like CDD or CWD) to behave unexpectedly.")
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            selected_month_names = st.multiselect("Select Months for Analysis", month_names, default=month_names)
            
            st.markdown("### 🔍 Raw Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
        if run_button:
            with st.spinner("Applying strict ETCCDI Quality Control..."):
                df_clean, qc_report = climate_core.apply_qc(df)
                
            month_map = {name: i+1 for i, name in enumerate(month_names)}
            selected_month_numbers = [month_map[m] for m in selected_month_names]
            if len(selected_month_numbers) < 12:
                df_clean = df_clean[df_clean['MONTH'].isin(selected_month_numbers)]
            
            with main_tab2:
                st.markdown("### Quality Control Summary")
                st.info("The dataset has been thoroughly scanned and cleaned to ensure the accuracy of the calculated indices.")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Missing Values (-99.9)", qc_report['missing_values'], help="Total number of missing or -99.9 placeholder values detected and removed.")
                col2.metric("Logical Errors", qc_report['logical_errors'], help="Days where the minimum temperature (TMIN) improperly exceeded the maximum temperature (TMAX).")
                col3.metric("Total Outliers Handled", sum([qc_report['outliers_tmax'], qc_report['outliers_tmin'], qc_report['outliers_prcp']]), help="Extreme values falling outside 3 standard deviations from the mean.")
                
                st.markdown("#### Outlier Breakdown (±3 Std Dev)")
                out_col1, out_col2, out_col3 = st.columns(3)
                out_col1.metric("TMAX Outliers", qc_report['outliers_tmax'])
                out_col2.metric("TMIN Outliers", qc_report['outliers_tmin'])
                out_col3.metric("PRCP Outliers", qc_report['outliers_prcp'])
                
            with st.spinner('🌍 Processing Climate Resolution... Please wait.'):
                results = cached_calculate_all_indices(
                    df_clean, hemisphere, thresh_fd, thresh_su, thresh_id, thresh_tr, thresh_rnn, resolution
                )
            
            with main_tab3:
                temp_indices = ['FD0', 'SU25', 'ID0', 'TR20', 'FD_user', 'SU_user', 'ID_user', 'TR_user', 'GSL', 'TXx', 'TNx', 'TXn', 'TNn', 'DTR', 'TN10p', 'TX10p', 'TN90p', 'TX90p', 'WSDI', 'CSDI', 'HWDI', 'HWI']
                precip_indices = ['R10mm', 'R20mm', 'R1mm', 'Rnn', 'CDD', 'CWD', 'PRCPTOT', 'RX1day', 'RX5day', 'SDII', 'R95p', 'R99p']
                seasonal_indices = ['MAM_TMAX_Ave', 'DJF_TMAX_Ave', 'JJAS_PRCP_Ave']
                
                sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["🌡️ Temperature", "🌧️ Precipitation", "🍂 Seasonal", "📋 Full Dataset"])
                with sub_tab1: st.dataframe(results[[c for c in temp_indices if c in results.columns]], use_container_width=True)
                with sub_tab2: st.dataframe(results[[c for c in precip_indices if c in results.columns]], use_container_width=True)
                with sub_tab3: st.dataframe(results[[c for c in seasonal_indices if c in results.columns]], use_container_width=True)
                with sub_tab4: st.dataframe(results, use_container_width=True)
                
                st.download_button("📥 Download Results", data=results.to_csv().encode('utf-8'), file_name="indices.csv", mime="text/csv")
                
            with main_tab4:
                thresholds_dict = {'SU25': "> 25°C", 'TR20': "> 20°C", 'FD0': "< 0°C", 'ID0': "< 0°C", 'SU_user': f"> {thresh_su}°C", 'TR_user': f"> {thresh_tr}°C", 'FD_user': f"< {thresh_fd}°C", 'ID_user': f"< {thresh_id}°C", 'Rnn': f">= {thresh_rnn}mm"}
                if not results.empty:
                    cols = st.columns(2)
                    for i, col_name in enumerate(results.columns):
                        with cols[i % 2]:
                            fig = create_plotly_figure(results, col_name, thresholds_dict, resolution, publication_mode)
                            
                            config_options = {
                                'displayModeBar': True,
                                'displaylogo': False,
                                'toImageButtonOptions': {
                                    'format': 'svg' if publication_mode else 'png',
                                    'filename': f'climatrend_{col_name}_{resolution}',
                                    'height': 800,
                                    'width': 1200,
                                    'scale': 1
                                }
                            }
                            st.plotly_chart(fig, use_container_width=True, config=config_options)

                            # --- Trend Statistics Row ---
                            slope, slope_se, r2, pval = compute_trend_stats(results[col_name])
                            if slope is not None:
                                sig_star = "" if pval >= 0.05 else ("*" if pval >= 0.01 else ("**" if pval >= 0.001 else "***"))
                                pval_color = "#16a34a" if pval < 0.05 else "#dc2626"
                                sig_label = "Significant" if pval < 0.05 else "Not Significant"
                                
                                r2_pct = r2 * 100
                                r2_comment = f"An R² of {r2:.4f} means the trend line represents a {r2_pct:.2f}% fit to the actual data points."
                                pval_pct = pval * 100
                                sig_text = "statistically significant" if pval < 0.05 else "not statistically significant"
                                pval_comment = f"A P-Value of {pval:.4f} means there is a {pval_pct:.2f}% chance this trend is just a random coincidence. At the standard 5% limit, this is {sig_text}."
                                
                                st.markdown(
                                    f"""
                                    <style>
                                    .tooltip-card {{
                                        position: relative; flex: 1; min-width: 110px; background: #f8fafc;
                                        border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px;
                                        text-align: center; cursor: help;
                                    }}
                                    .tooltip-card .tooltip-text {{
                                        visibility: hidden; width: 180px; background-color: #1e293b; color: #fff;
                                        text-align: center; border-radius: 6px; padding: 8px; position: absolute;
                                        z-index: 9999; bottom: 105%; left: 50%; margin-left: -90px;
                                        opacity: 0; transition: opacity 0.2s; font-size: 11px; font-family: Arial;
                                        box-shadow: 0px 4px 6px rgba(0,0,0,0.1); pointer-events: none; font-weight: normal;
                                    }}
                                    .tooltip-card:hover .tooltip-text {{ visibility: visible; opacity: 1; }}
                                    </style>
                                    <div style='display:flex; gap:10px; flex-wrap:wrap; margin:-10px 0 14px 0;'>
                                        <div class='tooltip-card'>
                                            <span class='tooltip-text'>{r2_comment}</span>
                                            <div style='font-size:11px; color:#64748b; font-family:Arial;'>R²</div>
                                            <div style='font-size:18px; font-weight:700; color:#1e3a8a; font-family:Arial;'>{r2:.4f}</div>
                                        </div>
                                        <div class='tooltip-card'>
                                            <span class='tooltip-text'>{pval_comment}</span>
                                            <div style='font-size:11px; color:#64748b; font-family:Arial;'>P-Value {sig_star}</div>
                                            <div style='font-size:18px; font-weight:700; color:{pval_color}; font-family:Arial;'>{pval:.4f}</div>
                                            <div style='font-size:10px; color:{pval_color}; font-family:Arial;'>{sig_label}</div>
                                        </div>
                                        <div class='tooltip-card'>
                                            <span class='tooltip-text'>Slope: The average change in this index per year based on linear regression.</span>
                                            <div style='font-size:11px; color:#64748b; font-family:Arial;'>Slope</div>
                                            <div style='font-size:18px; font-weight:700; color:#1e3a8a; font-family:Arial;'>{slope:.4f}</div>
                                            <div style='font-size:10px; color:#64748b; font-family:Arial;'>per year</div>
                                        </div>
                                        <div class='tooltip-card'>
                                            <span class='tooltip-text'>Slope Std Error: Measures the accuracy of the slope estimate (lower means more precise).</span>
                                            <div style='font-size:11px; color:#64748b; font-family:Arial;'>Slope Std Error</div>
                                            <div style='font-size:18px; font-weight:700; color:#1e3a8a; font-family:Arial;'>{slope_se:.4f}</div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

            with main_tab5:
                st.markdown("### 🌊 Return Period Analysis (GEV)")
                st.caption("Estimates the probability of extreme precipitation events using the Generalized Extreme Value distribution on Annual Maximum 1-Day Precipitation (RX1day).")
                st.info("💡 **How to read this:** A '1-in-50 Year' return period does not mean it happens exactly every 50 years. It means that **in any given year, there is exactly a 2% chance (1/50) that a catastrophic storm of that magnitude will occur.**")
                
                if 'RX1day' in results.columns and not results['RX1day'].empty:
                    rx1day_data = results['RX1day'].dropna()
                    if len(rx1day_data) > 10:
                        dist_choice = st.selectbox("📊 Select Statistical Distribution", ["Generalized Extreme Value (GEV)", "Gumbel (Type I Extreme Value)"])
                        return_periods = np.array([2, 5, 10, 20, 50, 100])
                        probabilities = 1 - 1 / return_periods
                        
                        if dist_choice == "Generalized Extreme Value (GEV)":
                            try:
                                params = sp_stats.genextreme.fit(rx1day_data)
                                return_levels = sp_stats.genextreme.ppf(probabilities, *params)
                            except Exception as e:
                                st.error(f"GEV fitting failed: {e}")
                                return_levels = None
                        else:
                            try:
                                params = sp_stats.gumbel_r.fit(rx1day_data)
                                return_levels = sp_stats.gumbel_r.ppf(probabilities, *params)
                            except Exception as e:
                                st.error(f"Gumbel fitting failed: {e}")
                                return_levels = None
                                
                        if return_levels is not None:
                            gev_fig = go.Figure()
                            gev_fig.add_trace(go.Scatter(
                                x=return_periods, y=return_levels,
                                mode='lines+markers', name='Return Level',
                                line=dict(color='#0ea5e9', width=3),
                                marker=dict(size=8, color='#0369a1'),
                                hovertemplate="Return Period: %{x} Years<br>Expected Max Rain: %{y:.2f} mm<extra></extra>"
                            ))
                            
                            gev_fig.update_layout(
                                title=dict(text="Precipitation Return Periods (RX1day)", font=dict(color='black')),
                                xaxis_title=dict(text="Return Period (Years)", font=dict(color='black')),
                                yaxis_title=dict(text="Expected Max Rainfall (mm)", font=dict(color='black')),
                                xaxis=dict(
                                    type='log', tickvals=[2, 5, 10, 20, 50, 100], ticktext=['2', '5', '10', '20', '50', '100'],
                                    showline=True, linewidth=2, linecolor='black', tickfont=dict(color='black')
                                ),
                                yaxis=dict(
                                    showline=True, linewidth=2, linecolor='black', tickfont=dict(color='black')
                                ),
                                font=dict(color='black'),
                                paper_bgcolor='#FFFFFF', plot_bgcolor='#f8fafc',
                                margin=dict(l=60, r=30, t=50, b=60)
                            )
                            st.plotly_chart(gev_fig, use_container_width=True)
                            
                            st.markdown("#### Estimated Return Levels")
                            cols = st.columns(len(return_periods))
                            for i, (rp, rl) in enumerate(zip(return_periods, return_levels)):
                                cols[i].metric(f"1-in-{rp} Year", f"{rl:.1f} mm")
                    else:
                        st.warning("Not enough data to fit GEV distribution (minimum 10 years required).")
                else:
                    st.warning("RX1day index not calculated. Cannot perform GEV analysis.")

            with main_tab6:
                st.markdown("### 🤖 AI Climate Research Assistant")
                st.caption("⚠️ Beta — responses may be inaccurate. Always verify scientific conclusions independently.")
                st.info("💡 Ask anything about your climate data in plain English. Powered by OpenAI GPT.")

                openai_key = st.text_input(
                    "🔑 Enter your OpenAI API Key",
                    type="password",
                    help="Your key is used only in this session and is never stored."
                )

                ai_target = st.radio(
                    "Ask questions about:",
                    ["Raw Climate Data (TMAX, TMIN, PRCP)", "Calculated ETCCDI Indices"],
                    horizontal=True
                )

                if openai_key:
                    # Build context dataframe
                    if ai_target == "Calculated ETCCDI Indices" and 'results' in dir() and not results.empty:
                        context_df = results.reset_index()
                        context_label = "ETCCDI climate indices"
                    else:
                        context_df = df[['YEAR', 'MONTH', 'DAY', 'TMAX', 'TMIN', 'PRCP']].head(500)
                        context_label = "daily climate station records"

                    # Show a preview
                    with st.expander(f"📊 Data context being sent to AI ({context_label})"):
                        st.dataframe(context_df.head(10), use_container_width=True)
                        st.caption(f"{len(context_df)} rows total (first 500 sent for context)")

                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = []

                    user_query = st.text_input(
                        "Ask your question:",
                        placeholder="e.g. 'Which year had the highest TXx?' or 'Summarize the rainfall trends'"
                    )

                    if user_query:
                        with st.spinner("🌍 Analyzing climate patterns..."):
                            try:
                                client = openai.OpenAI(api_key=openai_key)
                                csv_snippet = context_df.to_csv(index=False)
                                system_prompt = (
                                    "You are an expert climate scientist and data analyst. "
                                    "The user will provide a question about their climate dataset. "
                                    "Use the provided data to give a precise, scientific answer. "
                                    "Be concise but thorough. If calculations are needed, perform them."
                                )
                                user_prompt = (
                                    f"Here is the climate dataset ({context_label}):\n\n"
                                    f"{csv_snippet}\n\n"
                                    f"Question: {user_query}"
                                )
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user",   "content": user_prompt}
                                    ],
                                    max_tokens=1000,
                                    temperature=0.2
                                )
                                answer = response.choices[0].message.content
                                st.session_state.chat_history.append(
                                    {"question": user_query, "answer": answer}
                                )
                            except Exception as ai_err:
                                st.error(f"OpenAI Error: {str(ai_err)}")

                    if st.session_state.get("chat_history"):
                        st.markdown("#### 💬 Conversation")
                        for item in reversed(st.session_state.chat_history):
                            st.markdown(f"🧑‍🔬 **You:** {item['question']}")
                            st.markdown(f"🤖 **AI:** {item['answer']}")
                            st.divider()
                        if st.button("🗑️ Clear Chat"):
                            st.session_state.chat_history = []
                            st.rerun()
                else:
                    st.warning("⤴️ Enter your OpenAI API key above to start chatting with your climate data.")

            with main_tab7:
                st.markdown("### 🗺️ Station Map Visualizer")
                has_lat = 'LAT' in df_all_stations.columns or 'LATITUDE' in df_all_stations.columns
                has_lon = 'LON' in df_all_stations.columns or 'LONGITUDE' in df_all_stations.columns or 'LONG' in df_all_stations.columns

                if not (has_lat and has_lon):
                    st.info("📍 No geospatial columns detected in your dataset.\n\nTo use the map, add **LAT** and **LON** (or LATITUDE/LONGITUDE) columns to your CSV. Each row should have the coordinates of the weather station.")
                    st.code("YEAR,MONTH,DAY,TMAX,TMIN,PRCP,LAT,LON\n1985,1,1,28.5,18.2,0.0,23.7104,90.4074", language="csv")
                else:
                    lat_col = 'LAT' if 'LAT' in df_all_stations.columns else 'LATITUDE'
                    lon_col = 'LON' if 'LON' in df_all_stations.columns else ('LONGITUDE' if 'LONGITUDE' in df_all_stations.columns else 'LONG')
                    station_col = 'STATION' if 'STATION' in df_all_stations.columns else None

                    map_df = df_all_stations[[lat_col, lon_col] + ([station_col] if station_col else [])].drop_duplicates().dropna()
                    map_df.columns = ['lat', 'lon'] + (['station'] if station_col else [])

                    if station_col and 'selected_station' in locals():
                        map_df['color'] = np.where(map_df['station'] == selected_station, '#dc2626', '#2563eb')
                        map_df['size'] = np.where(map_df['station'] == selected_station, 18, 12)
                    else:
                        map_df['color'] = '#2563eb'
                        map_df['size'] = 14

                    fig_map = go.Figure(go.Scattermapbox(
                        lat=map_df['lat'], lon=map_df['lon'],
                        mode='markers',
                        marker=dict(size=map_df['size'], color=map_df['color'], opacity=0.9),
                        text=map_df['station'] if station_col else map_df['lat'].astype(str),
                        hovertemplate="<b>%{text}</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
                    ))

                    fig_map.update_layout(
                        mapbox_style="open-street-map",
                        mapbox=dict(
                            center=dict(lat=float(map_df['lat'].mean()), lon=float(map_df['lon'].mean())),
                            zoom=5
                        ),
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=550,
                        paper_bgcolor='#FFFFFF'
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                    st.caption(f"📌 {len(map_df)} station(s) detected. The currently selected station is highlighted in red.")

    except Exception as e:
        st.error(f"Critical Error during processing: {str(e)}")
else:
    st.info("👈 Upload your climate dataset in the sidebar to launch Climatrend.")
