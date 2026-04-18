import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import climate_core

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
    'FD': {'definition': 'Frost Days: Annual count of days where TMIN < threshold', 'unit': 'Days', 'significance': 'Important for agriculture, freezing impacts on infrastructure.'},
    'SU': {'definition': 'Summer Days: Annual count of days where TMAX > threshold', 'unit': 'Days', 'significance': 'Indicator of summer heat and potential heat stress.'},
    'ID': {'definition': 'Ice Days: Annual count of days where TMAX < threshold', 'unit': 'Days', 'significance': 'Measures duration of freezing conditions.'},
    'TR': {'definition': 'Tropical Nights: Annual count of days where TMIN > threshold', 'unit': 'Days', 'significance': 'High nighttime temperatures reduce relief from daytime heat.'},
    'GSL': {'definition': 'Growing Season Length', 'unit': 'Days', 'significance': 'Key metric for agricultural cycles and plant growth.'},
    'Rnn': {'definition': 'User-defined precipitation days', 'unit': 'Days', 'significance': 'Customizable extreme rainfall tracking.'},
    'CDD': {'definition': 'Consecutive Dry Days: Max consecutive days with PRCP < 1mm', 'unit': 'Days', 'significance': 'Indicator of drought risk and water scarcity.'},
    'CWD': {'definition': 'Consecutive Wet Days: Max consecutive days with PRCP >= 1mm', 'unit': 'Days', 'significance': 'Indicator of prolonged wet spells and potential flooding.'},
    'PRCPTOT': {'definition': 'Annual total PRCP from wet days', 'unit': 'mm', 'significance': 'Overall water availability from rainfall.'},
    'RX1day': {'definition': 'Annual maximum 1-day precipitation', 'unit': 'mm', 'significance': 'Measures extreme daily rainfall intensity.'},
    'RX5day': {'definition': 'Annual maximum consecutive 5-day precipitation', 'unit': 'mm', 'significance': 'Strongly correlated with flood risks over river basins.'},
    'SDII': {'definition': 'Simple Daily Intensity Index', 'unit': 'mm/day', 'significance': 'Average intensity of rainfall on wet days.'},
    'TXx': {'definition': 'Annual maximum of TMAX', 'unit': '°C', 'significance': 'The hottest day of the year.'},
    'TNx': {'definition': 'Annual maximum of TMIN', 'unit': '°C', 'significance': 'The warmest night of the year.'},
    'TXn': {'definition': 'Annual minimum of TMAX', 'unit': '°C', 'significance': 'The coldest day of the year.'},
    'TNn': {'definition': 'Annual minimum of TMIN', 'unit': '°C', 'significance': 'The coldest night of the year.'},
    'DTR': {'definition': 'Diurnal Temperature Range', 'unit': '°C', 'significance': 'Difference between daily max and min temperatures.'},
    'TN10p': {'definition': 'Cool nights: % of days where TMIN < 10th percentile', 'unit': '%', 'significance': 'Frequency of unusually cold nights.'},
    'TX10p': {'definition': 'Cool days: % of days where TMAX < 10th percentile', 'unit': '%', 'significance': 'Frequency of unusually cold days.'},
    'TN90p': {'definition': 'Warm nights: % of days where TMIN > 90th percentile', 'unit': '%', 'significance': 'Frequency of unusually warm nights.'},
    'TX90p': {'definition': 'Warm days: % of days where TMAX > 90th percentile', 'unit': '%', 'significance': 'Frequency of unusually warm days.'},
    'WSDI': {'definition': 'Warm Spell Duration Index', 'unit': 'Days', 'significance': 'Duration of prolonged heat waves.'},
    'CSDI': {'definition': 'Cold Spell Duration Index', 'unit': 'Days', 'significance': 'Duration of prolonged cold snaps.'},
    'R95p': {'definition': 'Very wet days: PRCP sum when PRCP > 95th percentile', 'unit': 'mm', 'significance': 'Contribution of extreme rainfall to total precipitation.'},
    'R99p': {'definition': 'Extremely wet days: PRCP sum when PRCP > 99th percentile', 'unit': 'mm', 'significance': 'Contribution of highly extreme rainfall.'},
    'MAM_TMAX_Ave': {'definition': 'Spring Average (MAM): Mean TMAX for March, April, May', 'unit': '°C', 'significance': 'Tracks springtime warming trends.'},
    'DJF_TMAX_Ave': {'definition': 'Winter Average (DJF): Mean TMAX for Dec (prev year), Jan, Feb', 'unit': '°C', 'significance': 'Tracks winter warming trends across the year boundary.'},
    'JJAS_PRCP_Ave': {'definition': 'Monsoon Average (JJAS): Mean daily PRCP for Jun, Jul, Aug, Sep', 'unit': 'mm/day', 'significance': 'Tracks monsoon/summer rainfall intensity.'}
}

def create_plotly_figure(df, col_name, thresholds):
    meta = INDEX_METADATA.get(col_name, {'definition': 'Index data', 'unit': '', 'significance': 'Climate variable'})
    
    thresh_val = thresholds.get(col_name)
    title_suffix = f" (Threshold: {thresh_val})" if thresh_val is not None else ""
    
    hover_text = (
        f"<b>Year:</b> %{{x}}<br>"
        f"<b>Value:</b> %{{y:.2f}} {meta['unit']}<br><br>"
        f"<b>Definition:</b> {meta['definition']}<br>"
        f"<b>Significance:</b> {meta['significance']}"
        f"<extra></extra>"
    )
    
    fig = go.Figure()
    valid_data = df[col_name].dropna()
    
    if valid_data.empty:
        return fig
    
    # Scatter points & connection lines
    fig.add_trace(go.Scatter(
        x=valid_data.index, 
        y=valid_data,
        mode='lines+markers',
        name='Actual Data',
        hovertemplate=hover_text,
        line=dict(color='#2563eb', width=2),
        marker=dict(size=6, color='#1e3a8a')
    ))
    
    # OLS Trendline
    if len(valid_data) > 1:
        z = np.polyfit(valid_data.index, valid_data, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=valid_data.index,
            y=p(valid_data.index),
            mode='lines',
            name='Trendline (OLS)',
            line=dict(color='#ef4444', width=2, dash='dash'),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f"<b>{col_name}{title_suffix} Trend</b>",
        xaxis_title="Year",
        yaxis_title=f"{col_name} ({meta['unit']})",
        margin=dict(l=20, r=20, t=40, b=20)
        # Removed hardcoded templates and background colors to fix Dark Mode visibility!
    )
    return fig

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

st.sidebar.markdown("### Location Settings")
hemisphere = st.sidebar.radio("Hemisphere (for GSL)", ["Northern", "Southern"], index=0)

st.sidebar.markdown("### Temperature Thresholds")
thresh_su = st.sidebar.number_input("Summer Day (SU) > °C", value=25.0)
thresh_tr = st.sidebar.number_input("Tropical Night (TR) > °C", value=20.0)
thresh_fd = st.sidebar.number_input("Frost Day (FD) < °C", value=0.0)
thresh_id = st.sidebar.number_input("Ice Day (ID) < °C", value=0.0)

st.sidebar.markdown("### Rainfall Thresholds")
thresh_rnn = st.sidebar.number_input("User-defined Rain (Rnn) >= mm", value=30.0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

# --- MAIN AREA ---
if uploaded_file is not None:
    try:
        df = climate_core.load_data(uploaded_file)
        
        main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
            "🔍 Data Preview", 
            "🛡️ QC Report", 
            "📊 Calculated Indices", 
            "📈 Professional Analytics Hub"
        ])
        
        with main_tab1:
            st.markdown("### Raw Dataset Head")
            st.dataframe(df.head(100), use_container_width=True)
            
        if run_button:
            with st.spinner("Applying strict ETCCDI Quality Control..."):
                df_clean, qc_report = climate_core.apply_qc(df)
            
            with main_tab2:
                st.markdown("### Quality Control Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Missing Values Handled", qc_report['missing_values'])
                col2.metric("Logical Errors Fixed", qc_report['logical_errors'], help="Tmax < Tmin")
                col3.metric("Outliers Handled", sum([qc_report['outliers_tmax'], qc_report['outliers_tmin'], qc_report['outliers_prcp']]))
                
                st.write("**Outlier Breakdown (>3 Std Dev):**")
                st.code(f"TMAX: {qc_report['outliers_tmax']} | TMIN: {qc_report['outliers_tmin']} | PRCP: {qc_report['outliers_prcp']}")

            with st.spinner("Crunching ETCCDI Indices, Percentiles, and Seasonal Averages..."):
                results = climate_core.calculate_all_indices(
                    df_clean, hemisphere=hemisphere,
                    thresh_fd=thresh_fd, thresh_su=thresh_su, thresh_id=thresh_id, thresh_tr=thresh_tr,
                    thresh_rnn=thresh_rnn
                )
            
            with main_tab3:
                st.markdown("### Complete Indices Matrix")
                
                temp_indices = ['FD', 'SU', 'ID', 'TR', 'GSL', 'TXx', 'TNx', 'TXn', 'TNn', 'DTR', 'TN10p', 'TX10p', 'TN90p', 'TX90p', 'WSDI', 'CSDI']
                precip_indices = ['Rnn', 'CDD', 'CWD', 'PRCPTOT', 'RX1day', 'RX5day', 'SDII', 'R95p', 'R99p']
                seasonal_indices = ['MAM_TMAX_Ave', 'DJF_TMAX_Ave', 'JJAS_PRCP_Ave']
                
                sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["🌡️ Temperature", "🌧️ Precipitation", "🍂 Seasonal", "📋 Full Dataset"])
                with sub_tab1: st.dataframe(results[[c for c in temp_indices if c in results.columns]], use_container_width=True)
                with sub_tab2: st.dataframe(results[[c for c in precip_indices if c in results.columns]], use_container_width=True)
                with sub_tab3: st.dataframe(results[[c for c in seasonal_indices if c in results.columns]], use_container_width=True)
                with sub_tab4: st.dataframe(results, use_container_width=True)
                
                csv_data = results.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Batch Download All Indices as CSV",
                    data=csv_data,
                    file_name="climatrend_results.csv",
                    mime="text/csv",
                    type="primary"
                )
                
            with main_tab4:
                st.markdown("### Interactive Visualizations")
                st.info("💡 **Pro Tip**: Hover over any point to read the scientific definition. Click the 📷 icon in the top right of the graph to download it as a PNG.")
                
                viz_tab1, viz_tab2 = st.tabs(["Focus Mode (Single Index)", "Grid View Dashboard"])
                
                thresholds_dict = {
                    'SU': f"> {thresh_su}°C",
                    'TR': f"> {thresh_tr}°C",
                    'FD': f"< {thresh_fd}°C",
                    'ID': f"< {thresh_id}°C",
                    'Rnn': f">= {thresh_rnn}mm"
                }
                
                with viz_tab1:
                    index_to_plot = st.selectbox("Select Index to Plot", results.columns)
                    if not results.empty and index_to_plot:
                        fig = create_plotly_figure(results, index_to_plot, thresholds_dict)
                        st.plotly_chart(fig, use_container_width=True, key=f"focus_plot_{index_to_plot}", config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'toImageButtonOptions': {'filename': f'{index_to_plot}_plot'}
                        })
                
                with viz_tab2:
                    if not results.empty:
                        cols = st.columns(2)
                        for i, col_name in enumerate(results.columns):
                            with cols[i % 2]:
                                fig = create_plotly_figure(results, col_name, thresholds_dict)
                                st.plotly_chart(fig, use_container_width=True, key=f"grid_plot_{col_name}_{i}", config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'toImageButtonOptions': {'filename': f'{col_name}_plot'}
                                })
                                
    except Exception as e:
        st.error(f"Critical Error during processing: {str(e)}")
else:
    st.info("👈 Upload your climate dataset in the sidebar to launch Climatrend.")
