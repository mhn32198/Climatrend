import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import climate_core
import subprocess, sys

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
    'GSL': {'definition': 'Growing Season Length: Length of the warm period suitable for plant growth', 'unit': 'Days', 'significance': 'Important for agriculture planning'},
    'TXx': {'definition': 'Max Tmax: Hottest day of the year', 'unit': '°C', 'significance': 'Measures peak heat intensity'},
    'TNn': {'definition': 'Min Tmin: Coldest night of the year', 'unit': '°C', 'significance': 'Identifies extreme cold events'},
    'TXn': {'definition': 'Min Tmax: Coldest daytime temperature', 'unit': '°C', 'significance': 'Shows severity of cold days'},
    'TNx': {'definition': 'Max Tmin: Warmest night of the year', 'unit': '°C', 'significance': 'Indicates heat stress at night'},
    'TN10p': {'definition': 'Cool Nights: Percentage of unusually cold nights', 'unit': '%', 'significance': 'Tracks cooling trends'},
    'TX10p': {'definition': 'Cool Days: Percentage of unusually cool days', 'unit': '%', 'significance': 'Measures cold variability'},
    'TN90p': {'definition': 'Warm Nights: Percentage of unusually warm nights', 'unit': '%', 'significance': 'Important for urban heat studies'},
    'TX90p': {'definition': 'Warm Days: Percentage of unusually hot days', 'unit': '%', 'significance': 'Tracks frequency of heatwaves'},
    'WSDI': {'definition': 'Warm Spell Duration: Number of days in long hot periods (heatwaves)', 'unit': 'Days', 'significance': 'Measures heatwave duration'},
    'CSDI': {'definition': 'Cold Spell Duration: Number of days in long cold periods', 'unit': 'Days', 'significance': 'Measures cold wave duration'},
    'DTR': {'definition': 'Diurnal Temperature Range: Difference between day and night temperature', 'unit': '°C', 'significance': 'Shows climate variability and urban effects'},
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
    
    hover_text = (
        f"<b>{col_name}</b><br>"
        f"Value: %{{y}} {meta['unit']}<br>"
        f"<i>{meta['definition']}</i><br>"
        f"<span style='font-size:0.9em; color:#666;'>{meta.get('significance', '')}</span>"
        "<extra></extra>"
    )
    
    line_color   = 'black'   if pub_mode else '#1d4ed8'
    marker_color = 'black'   if pub_mode else '#1e3a8a'
    trend_color  = '#555555' if pub_mode else '#dc2626'

    fig.add_trace(go.Scatter(
        x=valid_data.index, y=valid_data, mode='lines+markers',
        name='Observed',
        line=dict(color=line_color, width=2.5),
        marker=dict(size=6, color=marker_color, symbol='circle'),
        hovertemplate=hover_text
    ))
    
    if len(valid_data) > 1:
        x_numeric = np.arange(len(valid_data))
        z = np.polyfit(x_numeric, valid_data, 1)
        p = np.poly1d(z)
        slope, intercept = z[0], z[1]
        sign = '+' if intercept >= 0 else '-'
        formula_str = f"y = {slope:.3f}x {sign} {abs(intercept):.2f}"
        fig.add_trace(go.Scatter(
            x=valid_data.index, y=p(x_numeric), mode='lines',
            name='Linear Trend',
            line=dict(color=trend_color, width=2, dash='dash'),
            hoverinfo='skip'
        ))
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.01, y=0.01,
            text=f"<span style='font-size:9px; color:#888;'>{formula_str}</span>",
            showarrow=False,
            font=dict(size=9, color='#888888', family='Arial'),
            align='left'
        )
    
    fig.update_layout(
        title=None,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        margin=dict(l=60, r=30, t=30, b=60),
        font=dict(family='Arial, Helvetica, sans-serif', size=14, color='black'),
        xaxis=dict(
            title=dict(text=f"<b>{x_axis_title}</b>", font=dict(size=18, color='black', family='Arial')),
            tickfont=dict(size=14, color='black', family='Arial'),
            showline=True, linewidth=2.5, linecolor='black',
            mirror=True, ticks='outside', ticklen=6,
            showgrid=False, zeroline=False
        ),
        yaxis=dict(
            title=dict(text=f"<b>{col_name} ({meta['unit']})</b>", font=dict(size=18, color='black', family='Arial')),
            tickfont=dict(size=14, color='black', family='Arial'),
            showline=True, linewidth=2.5, linecolor='black',
            mirror=True, ticks='outside', ticklen=6,
            showgrid=False, zeroline=False
        ),
        legend=dict(
            x=0.98, y=0.98, xanchor='right', yanchor='top',
            bgcolor='white', bordercolor='black', borderwidth=1.5,
            font=dict(size=13, family='Arial', color='black')
        )
    )
    return fig

st.sidebar.header("⚙️ Configuration Panel")
uploaded_file = st.sidebar.file_uploader("Upload Climate Data (.csv or .txt)", type=["csv", "txt"])

st.sidebar.markdown("### Analysis Settings")
resolution = st.sidebar.radio("Analysis Resolution", ["Annual", "Monthly"], index=0)
publication_mode = st.sidebar.checkbox("🎓 Publication Mode (Journal Ready)", value=False, help="Removes titles, changes to high-contrast B&W, and enables SVG high-res export.")

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

if uploaded_file is not None:
    try:
        df = cached_load_data(uploaded_file)
        if 'STATION' in df.columns:
            st.sidebar.markdown("### Station Selection")
            stations = df['STATION'].unique()
            selected_station = st.sidebar.selectbox("🎯 Select Station", stations)
            df = df[df['STATION'] == selected_station].reset_index(drop=True)
            
        main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs([
            "🔍 Data Preview", "🛡️ QC Report", "📊 Calculated Indices",
            "📈 Professional Analytics Hub", "🤖 AI Research Assistant", "🗺️ Station Map"
        ])
        
        with main_tab1:
            st.dataframe(df.head(100), use_container_width=True)
            
        if run_button:
            with st.spinner("Applying strict ETCCDI Quality Control..."):
                df_clean, qc_report = climate_core.apply_qc(df)
            
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
                temp_indices = ['FD', 'SU', 'ID', 'TR', 'GSL', 'TXx', 'TNx', 'TXn', 'TNn', 'DTR', 'TN10p', 'TX10p', 'TN90p', 'TX90p', 'WSDI', 'CSDI']
                precip_indices = ['R10mm', 'R20mm', 'R1mm', 'Rnn', 'CDD', 'CWD', 'PRCPTOT', 'RX1day', 'RX5day', 'SDII', 'R95p', 'R99p']
                seasonal_indices = ['MAM_TMAX_Ave', 'DJF_TMAX_Ave', 'JJAS_PRCP_Ave']
                
                sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["🌡️ Temperature", "🌧️ Precipitation", "🍂 Seasonal", "📋 Full Dataset"])
                with sub_tab1: st.dataframe(results[[c for c in temp_indices if c in results.columns]], use_container_width=True)
                with sub_tab2: st.dataframe(results[[c for c in precip_indices if c in results.columns]], use_container_width=True)
                with sub_tab3: st.dataframe(results[[c for c in seasonal_indices if c in results.columns]], use_container_width=True)
                with sub_tab4: st.dataframe(results, use_container_width=True)
                
                st.download_button("📥 Download Results", data=results.to_csv().encode('utf-8'), file_name="indices.csv", mime="text/csv")
                
            with main_tab4:
                thresholds_dict = {'SU': f"> {thresh_su}°C", 'TR': f"> {thresh_tr}°C", 'FD': f"< {thresh_fd}°C", 'ID': f"< {thresh_id}°C", 'Rnn': f">= {thresh_rnn}mm"}
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
                                    'scale': 4
                                }
                            }
                            st.plotly_chart(fig, use_container_width=True, config=config_options)

            with main_tab5:
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

            with main_tab6:
                st.markdown("### 🗺️ Station Map Visualizer")
                has_lat = 'LAT' in df.columns or 'LATITUDE' in df.columns
                has_lon = 'LON' in df.columns or 'LONGITUDE' in df.columns or 'LONG' in df.columns

                if not (has_lat and has_lon):
                    st.info("📍 No geospatial columns detected in your dataset.\n\nTo use the map, add **LAT** and **LON** (or LATITUDE/LONGITUDE) columns to your CSV. Each row should have the coordinates of the weather station.")
                    st.code("YEAR,MONTH,DAY,TMAX,TMIN,PRCP,LAT,LON\n1985,1,1,28.5,18.2,0.0,23.7104,90.4074", language="csv")
                else:
                    lat_col = 'LAT' if 'LAT' in df.columns else 'LATITUDE'
                    lon_col = 'LON' if 'LON' in df.columns else ('LONGITUDE' if 'LONGITUDE' in df.columns else 'LONG')

                    station_col = 'STATION' if 'STATION' in df.columns else None
                    map_df = df[[lat_col, lon_col] + ([station_col] if station_col else [])].drop_duplicates().dropna()
                    map_df.columns = ['lat', 'lon'] + (['station'] if station_col else [])

                    map_col_option = None
                    if 'results' in dir() and not results.empty:
                        map_col_option = st.selectbox("🎨 Color-code stations by index value (optional):", ["None"] + list(results.columns))

                    hover_name = map_df['station'] if station_col else None

                    if map_col_option and map_col_option != "None" and station_col and station_col in df.columns:
                        # Merge index value per station
                        idx_per_station = df.groupby(station_col).first()[[lat_col, lon_col]].reset_index()
                        fig_map = go.Figure(go.Scattermapbox(
                            lat=map_df['lat'], lon=map_df['lon'],
                            mode='markers',
                            marker=dict(size=14, color='#2563eb', opacity=0.85),
                            text=map_df.get('station', ''),
                            hovertemplate="<b>%{text}</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
                        ))
                    else:
                        fig_map = go.Figure(go.Scattermapbox(
                            lat=map_df['lat'], lon=map_df['lon'],
                            mode='markers',
                            marker=dict(size=14, color='#2563eb', opacity=0.85),
                            text=map_df.get('station', map_df['lat'].astype(str)) if station_col else map_df['lat'].astype(str),
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
                    st.caption(f"📌 {len(map_df)} station(s) detected from geospatial columns `{lat_col}` / `{lon_col}`.")

    except Exception as e:
        st.error(f"Critical Error during processing: {str(e)}")
else:
    st.info("👈 Upload your climate dataset in the sidebar to launch Climatrend.")
