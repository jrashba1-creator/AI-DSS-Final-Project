import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium, folium_static
import plotly.express as px
from pathlib import Path

# Import custom class
from utils import LogTransformedRegressor

def classify_alkalinity_risk(value):
    """Classify alkalinity based on WHO/SANS 241 standards"""
    if value < 20:
        return 'Very Low', 'orange', '⚠️'
    elif value < 75:
        return 'Low', 'yellow', '⚠️'
    elif value <= 200:
        return 'Normal', 'green', '✓'
    elif value <= 400:
        return 'High', 'orange', '⚠️'
    else:
        return 'Very High', 'red', '🔴'

def classify_ec_risk(value):
    """Classify EC based on WHO/SANS 241 standards"""
    if value < 70:
        return 'Excellent', 'green', '✓'
    elif value <= 300:
        return 'Good', 'green', '✓'
    elif value <= 600:
        return 'Moderate', 'orange', '⚠️'
    else:
        return 'High', 'red', '🔴'

def classify_drp_risk(value):
    """Classify DRP based on WHO/SANS 241 standards"""
    if value < 50:
        return 'Safe', 'green', '✓'
    elif value < 100:
        return 'Warning', 'orange', '⚠️'
    else:
        return 'Severe', 'red', '🔴'

def get_overall_site_risk(alk_color, ec_color, drp_color):
    """Determine overall site marker color - worst case wins"""
    colors = [alk_color, ec_color, drp_color]
    if 'red' in colors:
        return 'red'
    elif 'orange' in colors:
        return 'orange'
    else:
        return 'green'


# Page configuration
st.set_page_config(
    page_title="Water Quality Prediction Dashboard",
    page_icon="💧",
    layout="wide"
)

# Title
# Centered Title (larger)
st.markdown("<h1 style='text-align: center; font-size: 45px;'>💧 South Africa Water Quality Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["🗺️ Site Predictions", "📤 Upload Custom Data"])

# Load models (cached so it only loads once)
@st.cache_resource
def load_models():
    models = {}
    for target in ['Alkalinity', 'EC', 'DRP']:
        filepath = f'dashboard/models/{target.lower()}_model.pkl'  # Added 'dashboard/'
        
        # Load with explicit file opening
        with open(filepath, 'rb') as f:
            data = joblib.load(f)
        
        # Wrap back in LogTransformedRegressor if needed
        if data.get('is_log_transformed', False):
            from utils import LogTransformedRegressor
            wrapped_model = LogTransformedRegressor(data['model'])
            wrapped_model.model_ = data['model']
            data['model'] = wrapped_model
        
        models[target] = data
    return models
# Load test data (cached)
@st.cache_data
def load_test_data():
    df = pd.read_csv('dashboard/ml_ready_test.csv')  # Added 'dashboard/'
    # Add Cluster column if missing
    if 'Cluster' not in df.columns:
        df['Cluster'] = 0
    return df

# Load models and data
try:
    models = load_models()
    test_data = load_test_data()
    st.sidebar.success(f"✅ Models loaded successfully")
    st.sidebar.info(f"📊 Test data: {len(test_data)} samples from {test_data['Location_ID'].nunique()} sites")
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# Create tabs


# ============================================
# TAB 1: SITE PREDICTIONS MAP
# ============================================

with tab1:
    st.markdown("""
    <div style='font-size: 20px;'>
    This dashboard uses machine learning models trained on water quality 
    data from 162 monitoring sites across South Africa (2011-2014).
    <br><br>
    <b>Models predict three parameters:</b>
    <ul style='font-size: 18px;'>
    <li>Total Alkalinity (mg/L)</li>
    <li>Electrical Conductance (mS/m)</li>
    <li>Dissolved Reactive Phosphorus (pollution risk)</li>
    </ul>
    The map shows predictions for 148 sites where 2015 test data is available.
    </div>
    """, unsafe_allow_html=True)

    st.header("Interactive Site Map")
    # Get latest sample for each site
    test_data['Sample_Date'] = pd.to_datetime(test_data['Sample Date'], format='mixed', dayfirst=True)
    latest_data = test_data.sort_values('Sample_Date').groupby('Location_ID').tail(1).reset_index(drop=True)
    
    st.write(f"Showing predictions for **{len(latest_data)} sites** (latest 2015 samples)")
    
    # Make predictions for all sites
    predictions = {}
    for target in ['Alkalinity', 'EC', 'DRP']:
        model_info = models[target]
        model = model_info['model']
        features = model_info['features']
    
        X = latest_data[features]
        preds = model.predict(X)
        predictions[target] = preds

    # Add predictions to dataframe
    latest_data['pred_Alkalinity'] = predictions['Alkalinity']
    latest_data['pred_EC'] = predictions['EC']
    latest_data['pred_DRP'] = predictions['DRP']

    # Classify all parameters using WHO/SANS 241 standards
    latest_data[['Alk_Risk', 'Alk_Color', 'Alk_Icon']] = latest_data['pred_Alkalinity'].apply(
        lambda x: pd.Series(classify_alkalinity_risk(x))
    )
    latest_data[['EC_Risk', 'EC_Color', 'EC_Icon']] = latest_data['pred_EC'].apply(
        lambda x: pd.Series(classify_ec_risk(x))
    )
    latest_data[['DRP_Risk', 'DRP_Color', 'DRP_Icon']] = latest_data['pred_DRP'].apply(
        lambda x: pd.Series(classify_drp_risk(x))
    )

    # Overall site risk (worst parameter determines marker color)
    latest_data['Site_Color'] = latest_data.apply(
        lambda row: get_overall_site_risk(row['Alk_Color'], row['EC_Color'], row['DRP_Color']), 
        axis=1
    )

    # Create map centered on South Africa
    m = folium.Map(
        location=[-29.0, 24.0],
        zoom_start=5,
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google Hybrid'
    )
    
    # Add markers for each site

    for idx, row in latest_data.iterrows():
        # Color based on overall site risk
        color = row['Site_Color']
        
        # Create popup text with risk classifications
        popup_text = f"""
        <b>Location ID:</b> {row['Location_ID']}<br>
        <b>Date:</b> {row['Sample_Date'].strftime('%Y-%m-%d')}<br>
        <hr>
        <b>Alkalinity:</b> {row['pred_Alkalinity']:.2f} mg/L 
        <span style="color:{row['Alk_Color']}">{row['Alk_Icon']} {row['Alk_Risk']}</span><br>
        <b>EC:</b> {row['pred_EC']:.2f} mS/m 
        <span style="color:{row['EC_Color']}">{row['EC_Icon']} {row['EC_Risk']}</span><br>
        <b>DRP:</b> {row['pred_DRP']:.2f} μg/L 
        <span style="color:{row['DRP_Color']}">{row['DRP_Icon']} {row['DRP_Risk']}</span><br>
        """
    
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"Site {row['Location_ID']} - Overall: {row['Site_Color'].upper()}",
            icon=folium.Icon(color=color, icon='tint', prefix='fa')
        ).add_to(m)
    
    # Display legend and map side by side
    col_map, col_legend = st.columns([3, 1])

    with col_map:
        folium_static(m, width=2000, height=600)

    with col_legend:
        st.markdown("### 📊 Water Quality Standards")
        
        st.markdown("**DRP (μg/L):**")
        st.markdown("🟢 Safe: < 50")
        st.markdown("🟠 Warning: 50-99")
        st.markdown("🔴 Severe: ≥ 100")
        
        st.markdown("**Alkalinity (mg/L):**")
        st.markdown("🟠 Very Low: < 20")
        st.markdown("🟡 Low: 20-74")
        st.markdown("🟢 Normal: 75-200")
        st.markdown("🟠 High: 201-400")
        st.markdown("🔴 Very High: > 400")
        
        st.markdown("**EC (mS/m):**")
        st.markdown("🟢 Excellent: < 70")
        st.markdown("🟢 Good: 70-300")
        st.markdown("🟠 Moderate: 301-600")
        st.markdown("🔴 High: > 600")
        
        st.caption("*Based on WHO & SANS 241 standards*")
    
    # Summary statistics
 
    st.subheader("📊 Risk Summary")
    
    
    col1, col2, col3, col4 = st.columns(4)
    # Add custom CSS for larger metric values
    st.markdown("""
    <style>
        [data-testid="stMetricValue"] {
            font-size: 70px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 60px;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        red_sites = (latest_data['Site_Color'] == 'red').sum()
        st.metric("🔴 High Risk Sites", red_sites, 
                delta=f"{(red_sites/len(latest_data)*100):.1f}%")

    with col2:
        orange_sites = (latest_data['Site_Color'] == 'orange').sum()
        st.metric("⚠️ Warning Sites", orange_sites,
                delta=f"{(orange_sites/len(latest_data)*100):.1f}%")

    with col3:
        green_sites = (latest_data['Site_Color'] == 'green').sum()
        st.metric("✓ Safe Sites", green_sites,
                delta=f"{(green_sites/len(latest_data)*100):.1f}%")

    with col4:
        st.metric("Total Sites", len(latest_data))

    # ADD THIS NEW SECTION HERE:
    st.subheader("📈 Prediction Distributions")

    # Create three columns for the histograms
    hist_col1, hist_col2, hist_col3 = st.columns(3)

    with hist_col1:
        # Alkalinity histogram
        fig_alk = px.histogram(
            latest_data, 
            x='pred_Alkalinity',
            color='Alk_Risk',
            title='Alkalinity Distribution',
            labels={'pred_Alkalinity': 'Alkalinity (mg/L)', 'count': 'Number of Sites'},
            color_discrete_map={
                'Very Low': 'orange',
                'Low': '#DAA520',
                'Normal': 'green',
                'High': 'orange',
                'Very High': 'red'
            },
            nbins=30
        )
        fig_alk.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_alk, use_container_width=True)

    with hist_col2:
        # EC histogram
        fig_ec = px.histogram(
            latest_data,
            x='pred_EC',
            color='EC_Risk',
            title='Electrical Conductance Distribution',
            labels={'pred_EC': 'EC (mS/m)', 'count': 'Number of Sites'},
            color_discrete_map={
                'Excellent': 'green',
                'Good': 'lightgreen',
                'Moderate': 'orange',
                'High': 'red'
            },
            nbins=30
        )
        fig_ec.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_ec, use_container_width=True)

    with hist_col3:
        # DRP histogram
        fig_drp = px.histogram(
            latest_data,
            x='pred_DRP',
            color='DRP_Risk',
            title='DRP Distribution',
            labels={'pred_DRP': 'DRP (μg/L)', 'count': 'Number of Sites'},
            color_discrete_map={
                'Safe': 'green',
                'Warning': 'orange',
                'Severe': 'red'
            },
            nbins=30
        )
        fig_drp.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_drp, use_container_width=True)
    # Distribution histograms (existing code above)

    # Model Info Expander
    # OPTION 4: Model Info Expander
    with st.expander("ℹ️ About the Models"):
        st.markdown("""
        <div style='font-size: 20px;'>
        
        <p><b>Model Performance (2015 Test Set):</b></p>
        <ul style='font-size: 18px;'>
        <li><b>Alkalinity:</b> LightGBM Regressor (R² = 0.70, 65 features)</li>
        <li><b>EC:</b> XGBoost Regressor (R² = 0.78, 72 features)</li>
        <li><b>DRP:</b> XGBoost Regressor (62% recall, 63% precision, 66 features)</li>
        </ul>
        
        <p style='font-size: 20px;'><b>Training Data:</b> 2011-2014 samples from 145 monitoring sites</p>
        
        <p style='font-size: 20px;'><b>Features Include:</b></p>
        <ul style='font-size: 20px;'>
        <li>Climate data (temperature, precipitation, soil moisture)</li>
        <li>Satellite imagery (NDVI, NBR, NDBI, land cover percentages)</li>
        <li>Temporal patterns (rolling averages, lags)</li>
        <li>Spatial clustering (KMeans grouping)</li>
        <li>Static features (elevation, coordinates)</li>
        </ul>
        
        <p style='font-size: 20px;'><b>Risk Thresholds:</b> Based on WHO drinking water guidelines and South African SANS 241 standards</p>
        
        </div>
        """, unsafe_allow_html=True)

    # OPTION 2: Top 10 Worst Sites
    st.subheader("⚠️ Top 10 Sites Requiring Attention")

    # Create risk score: red=3, orange=2, yellow=2, green=1
    risk_scores = {
        'red': 3,
        'orange': 2,
        'yellow': 2,
        'green': 1
    }

    latest_data['risk_score'] = (
        latest_data['Alk_Color'].map(risk_scores).fillna(1) +
        latest_data['EC_Color'].map(risk_scores).fillna(1) +
        latest_data['DRP_Color'].map(risk_scores).fillna(1)
    )

    worst_sites = latest_data.nlargest(10, 'risk_score')[
        ['Location_ID', 'pred_Alkalinity', 'Alk_Risk', 'pred_EC', 'EC_Risk', 'pred_DRP', 'DRP_Risk', 'risk_score']
    ].copy()

    # Format the table for better display
    worst_sites_display = worst_sites.rename(columns={
        'Location_ID': 'Site ID',
        'pred_Alkalinity': 'Alkalinity (mg/L)',
        'Alk_Risk': 'Alk Status',
        'pred_EC': 'EC (mS/m)',
        'EC_Risk': 'EC Status',
        'pred_DRP': 'DRP (μg/L)',
        'DRP_Risk': 'DRP Status',
        'risk_score': 'Risk Score'
    })

    # Round numeric columns
    worst_sites_display['Alkalinity (mg/L)'] = worst_sites_display['Alkalinity (mg/L)'].round(2)
    worst_sites_display['EC (mS/m)'] = worst_sites_display['EC (mS/m)'].round(2)
    worst_sites_display['DRP (μg/L)'] = worst_sites_display['DRP (μg/L)'].round(2)

    st.dataframe(worst_sites_display, use_container_width=True, hide_index=True)

    st.caption("*Risk Score: Sum of individual parameter risks (Red=3, Orange/Yellow=2, Green=1). Higher scores indicate sites needing immediate attention.*")

# ============================================
# TAB 2: UPLOAD CUSTOM DATA
# ============================================
with tab2:
    st.header("Upload Custom Predictions")
    st.markdown("Upload a CSV file with pre-computed features for the monitoring sites.")
    
    # Download template
    if st.button("📥 Download CSV Template"):
        # Get required features from models
        all_features = set()
        for target in ['Alkalinity', 'EC', 'DRP']:
            all_features.update(models[target]['features'])
        
        required_cols = ['Location_ID', 'Latitude', 'Longitude', 'Sample Date'] + sorted(all_features)
        template_df = pd.DataFrame(columns=required_cols)
        
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="Download Template",
            data=csv,
            file_name="water_quality_template.csv",
            mime="text/csv"
        )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load uploaded data
            uploaded_data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded: {len(uploaded_data)} rows")
            
            # Validate required columns
            required_base = ['Location_ID', 'Latitude', 'Longitude', 'Sample Date']
            missing_base = [col for col in required_base if col not in uploaded_data.columns]
            
            if missing_base:
                st.error(f"❌ Missing required columns: {missing_base}")
            else:
                # Check if Location_IDs are valid
                valid_locations = test_data['Location_ID'].unique()
                invalid_locs = uploaded_data[~uploaded_data['Location_ID'].isin(valid_locations)]['Location_ID'].unique()
                
                if len(invalid_locs) > 0:
                    st.warning(f"⚠️ Found {len(invalid_locs)} invalid Location_IDs (not in known sites)")
                    st.write("Invalid IDs:", invalid_locs[:10])
                
                # Make predictions
                uploaded_predictions = {}
                for target in ['Alkalinity', 'EC', 'DRP']:
                    model_info = models[target]
                    model = model_info['model']
                    features = model_info['features']
                    
                    # Check if all features present
                    missing_features = [f for f in features if f not in uploaded_data.columns]
                    if missing_features:
                        st.error(f"❌ {target} model missing features: {missing_features[:5]}...")
                        continue
                    
                    X = uploaded_data[features]
                    preds = model.predict(X)
                    uploaded_predictions[target] = preds
                
                if len(uploaded_predictions) == 3:
                    # Add predictions to dataframe
                    uploaded_data['pred_Alkalinity'] = uploaded_predictions['Alkalinity']
                    uploaded_data['pred_EC'] = uploaded_predictions['EC']
                    uploaded_data['pred_DRP'] = uploaded_predictions['DRP']
                    
                    # Classify all parameters using WHO/SANS 241 standards
                    uploaded_data[['Alk_Risk', 'Alk_Color', 'Alk_Icon']] = uploaded_data['pred_Alkalinity'].apply(
                        lambda x: pd.Series(classify_alkalinity_risk(x))
                    )
                    uploaded_data[['EC_Risk', 'EC_Color', 'EC_Icon']] = uploaded_data['pred_EC'].apply(
                        lambda x: pd.Series(classify_ec_risk(x))
                    )
                    uploaded_data[['DRP_Risk', 'DRP_Color', 'DRP_Icon']] = uploaded_data['pred_DRP'].apply(
                        lambda x: pd.Series(classify_drp_risk(x))
                    )
                    
                    # Overall site risk
                    uploaded_data['Site_Color'] = uploaded_data.apply(
                        lambda row: get_overall_site_risk(row['Alk_Color'], row['EC_Color'], row['DRP_Color']), 
                        axis=1
                    )
                    
                    st.success("✅ Predictions generated successfully!")
                    
                    # Show preview
                    st.subheader("Preview Results")
                    display_cols = ['Location_ID', 'Sample Date', 'pred_Alkalinity', 'Alk_Risk', 
                                    'pred_EC', 'EC_Risk', 'pred_DRP', 'DRP_Risk']
                    st.dataframe(uploaded_data[display_cols].head(10))
                    
                    # Download results
                    result_csv = uploaded_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=result_csv,
                        file_name="water_quality_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Show map with uploaded data
                    st.subheader("🗺️ Uploaded Data Map")
                    
                    # Get latest per site from uploaded data
                    uploaded_data['Sample_Date'] = pd.to_datetime(uploaded_data['Sample Date'], format='mixed', dayfirst=True)
                    latest_uploaded = uploaded_data.sort_values('Sample_Date').groupby('Location_ID').tail(1)
                    
                    # Display legend and map side by side
                    col_map2, col_legend2 = st.columns([3, 1])
                    
                    with col_map2:
                        m2 = folium.Map(
                            location=[-29.0, 24.0],
                            zoom_start=5,
                            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                            attr='Google Hybrid'
                        )
                        
                        for idx, row in latest_uploaded.iterrows():
                            color = row['Site_Color']
                            
                            popup_text = f"""
                            <b>Location ID:</b> {row['Location_ID']}<br>
                            <b>Date:</b> {row['Sample_Date'].strftime('%Y-%m-%d')}<br>
                            <hr>
                            <b>Alkalinity:</b> {row['pred_Alkalinity']:.2f} mg/L 
                            <span style="color:{row['Alk_Color']}">{row['Alk_Icon']} {row['Alk_Risk']}</span><br>
                            <b>EC:</b> {row['pred_EC']:.2f} mS/m 
                            <span style="color:{row['EC_Color']}">{row['EC_Icon']} {row['EC_Risk']}</span><br>
                            <b>DRP:</b> {row['pred_DRP']:.2f} μg/L 
                            <span style="color:{row['DRP_Color']}">{row['DRP_Icon']} {row['DRP_Risk']}</span><br>
                            """
                            
                            folium.Marker(
                                location=[row['Latitude'], row['Longitude']],
                                popup=folium.Popup(popup_text, max_width=300),
                                tooltip=f"Site {row['Location_ID']} - Overall: {row['Site_Color'].upper()}",
                                icon=folium.Icon(color=color, icon='tint', prefix='fa')
                            ).add_to(m2)
                        
                        folium_static(m2, width=1000, height=700)
                    
                    with col_legend2:
                        st.markdown("### 📊 Water Quality Standards")
                        st.markdown("**DRP (μg/L):**")
                        st.markdown("🟢 Safe: < 50")
                        st.markdown("🟠 Warning: 50-99")
                        st.markdown("🔴 Severe: ≥ 100")
                        st.markdown("**Alkalinity (mg/L):**")
                        st.markdown("🟠 Very Low: < 20")
                        st.markdown("🟡 Low: 20-74")
                        st.markdown("🟢 Normal: 75-200")
                        st.markdown("🟠 High: 201-400")
                        st.markdown("🔴 Very High: > 400")
                        st.markdown("**EC (mS/m):**")
                        st.markdown("🟢 Excellent: < 70")
                        st.markdown("🟢 Good: 70-300")
                        st.markdown("🟠 Moderate: 301-600")
                        st.markdown("🔴 High: > 600")
                        st.caption("*Based on WHO & SANS 241 standards*")
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")