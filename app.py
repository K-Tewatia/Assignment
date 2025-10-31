# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple
import csv  # Add this with other imports at the top
from io import BytesIO, StringIO  # Add StringIO
import time
# Import custom modules
from src.config import Config
from src.pinecone_manager import PineconeManager
from src.mistral_agent import MistralAgent
from src.data_processor import DataProcessor
from src.optimizer import BudgetOptimizer

# Page configuration
st.set_page_config(
    page_title="Marketing ROI Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 5px;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #165a8a, #1f7a1f);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'optimization_complete' not in st.session_state:
        st.session_state.optimization_complete = False
    if 'marketing_data' not in st.session_state:
        st.session_state.marketing_data = None
    if 'channel_metrics' not in st.session_state:
        st.session_state.channel_metrics = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

init_session_state()

# Initialize components
@st.cache_resource
def init_components() -> Tuple[Optional[PineconeManager], Optional[MistralAgent], Optional[DataProcessor]]:
    """Initialize components with proper error handling"""
    try:
        pinecone_key = Config.get_pinecone_api_key()
        mistral_key = Config.get_mistral_api_key()
        
        pinecone_mgr = PineconeManager(
            api_key=pinecone_key,
            index_name=Config.get_pinecone_index_name()
        )
        
        mistral_agent = MistralAgent(
            api_key=mistral_key,
            model=Config.get_mistral_model()
        )
        
        data_processor = DataProcessor()
        
        return pinecone_mgr, mistral_agent, data_processor
        
    except ValueError as e:
        st.error(f"❌ Configuration Error: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Initialization Error: {str(e)}")
        return None, None, None

# Helper functions
def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.2f}%"

def format_context_for_mistral(channel_metrics):
    """Format channel metrics as context for Mistral"""
    context = "CHANNEL PERFORMANCE METRICS:\n\n"
    
    for channel, metrics in channel_metrics.items():
        context += f"""{channel}:
  - Total Cost: ${metrics['cost']:,.2f}
  - Total Revenue: ${metrics['revenue']:,.2f}
  - Total Conversions: {metrics['conversions']:,}
  - Total Clicks: {metrics['clicks']:,}
  - Total Impressions: {metrics['impressions']:,}
  - ROI: {metrics['roi']:.2f}%
  - CPA: ${metrics['cpa']:.2f}
  - CTR: {metrics['ctr']:.2f}%
  - Conversion Rate: {metrics['conversion_rate']:.2f}%
  - Number of Campaigns: {metrics['num_campaigns']}

"""
    return context

def generate_sample_data():
    """Generate sample marketing data with new schema"""
    channels = ['Email', 'Social Media', 'Paid Search', 'Organic Search', 'Display Ads']
    campaign_names = {
        'Email': ['Newsletter Q1', 'Promo Blast', 'Welcome Series'],
        'Social Media': ['Facebook Campaign', 'Instagram Ads', 'LinkedIn B2B'],
        'Paid Search': ['Google Search Ads', 'Bing Ads'],
        'Organic Search': ['SEO Campaign'],
        'Display Ads': ['Banner Ads', 'Retargeting']
    }
    
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    
    data = []
    campaign_id_counter = 1000
    
    for date in dates:
        for channel in channels:
            for campaign_name in campaign_names[channel]:
                if channel == 'Email':
                    cost = np.random.uniform(400, 600)
                    roi_multiplier = np.random.uniform(30, 40)
                    impressions = int(np.random.uniform(8000, 12000))
                    ctr = np.random.uniform(8, 12)
                elif channel == 'Social Media':
                    cost = np.random.uniform(1200, 1800)
                    roi_multiplier = np.random.uniform(2.5, 4)
                    impressions = int(np.random.uniform(80000, 120000))
                    ctr = np.random.uniform(3, 6)
                elif channel == 'Paid Search':
                    cost = np.random.uniform(1500, 2500)
                    roi_multiplier = np.random.uniform(1.8, 2.5)
                    impressions = int(np.random.uniform(120000, 180000))
                    ctr = np.random.uniform(4, 7)
                elif channel == 'Organic Search':
                    cost = np.random.uniform(800, 1200)
                    roi_multiplier = np.random.uniform(18, 25)
                    impressions = int(np.random.uniform(40000, 60000))
                    ctr = np.random.uniform(5, 8)
                else:  # Display Ads
                    cost = np.random.uniform(1000, 1500)
                    roi_multiplier = np.random.uniform(1.2, 2)
                    impressions = int(np.random.uniform(100000, 150000))
                    ctr = np.random.uniform(0.5, 2)
                
                revenue = cost * roi_multiplier
                clicks = int(impressions * (ctr / 100))
                conversions = int(clicks * np.random.uniform(0.02, 0.08))
                conversion_rate = (conversions / clicks * 100) if clicks > 0 else 0
                roi = ((revenue - cost) / cost * 100) if cost > 0 else 0
                
                data.append({
                    'campaign_id': f'CMP-{campaign_id_counter}',
                    'campaign_name': campaign_name,
                    'channel': channel,
                    'date': date.strftime('%Y-%m-%d'),
                    'impressions': impressions,
                    'clicks': clicks,
                    'conversions': conversions,
                    'conversion_rate': round(conversion_rate, 2),
                    'cost': round(cost, 2),
                    'revenue': round(revenue, 2),
                    'roi': round(roi, 2),
                    'currency': 'USD'
                })
                
                campaign_id_counter += 1
    
    return pd.DataFrame(data)

# Main application
def main():
    # Validate configuration
    is_valid, errors = Config.validate_config()
    
    if not is_valid:
        st.error("⚠️ **Configuration Errors Detected!**")
        st.markdown("### Missing Configuration:")
        for error in errors:
            st.markdown(f"- ❌ {error}")
        
        st.markdown("""
        ### Quick Setup Guide:
        
        1. **Create `.env` file** in your project root:
        ```
        touch .env
        ```
        
        2. **Add your API keys**:
        ```
        PINECONE_API_KEY=your_pinecone_key_here
        MISTRAL_API_KEY=your_mistral_key_here
        ```
        
        3. **Optional settings**:
        ```
        PINECONE_INDEX_NAME=marketing-roi-optimizer
        MISTRAL_MODEL=mistral-large-latest
        ```
        
        4. **Restart the application**
        """)
        st.stop()
    
    # Initialize components
    pinecone_mgr, mistral_agent, data_processor = init_components()
    
    if pinecone_mgr is None or mistral_agent is None:
        st.error("❌ Failed to initialize components.")
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Marketing ROI Optimizer with AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #666; margin-bottom: 2rem;">Includes Pinecone data + Mistral AI to generate responses</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Home", "📥 Data Input", "📊 AI Analysis", "💰 Budget Optimization", "📄 Generate Report", "❓ Q&A Chat"]
    )
    
    # System status in sidebar
    st.sidebar.markdown("---")
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📥 Data Input":
        show_data_input_page(pinecone_mgr, data_processor)
    elif page == "📊 AI Analysis":
        show_analysis_page(pinecone_mgr, mistral_agent)
    elif page == "💰 Budget Optimization":
        show_optimization_page(pinecone_mgr, mistral_agent)
    elif page == "📄 Generate Report":
        show_report_page(pinecone_mgr, mistral_agent)
    elif page == "❓ Q&A Chat":
        show_qa_page(pinecone_mgr, mistral_agent)

def show_home_page():
    """Home page with overview and instructions"""
    
    st.markdown("---")
    
    # Features
    st.markdown("## 🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📥 Data Management
        - **CSV Upload**: Import your marketing data
        - **Sample Data**: Generate test data instantly
        - **Data Validation**: Automatic quality checks
        - **Pinecone Storage**: Vector database integration
        - **Multi-Campaign Support**: Handle multiple campaigns
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI Analysis
        - **ROI Analysis**: Calculate channel performance
        - **Benchmark Comparison**: Industry standards
        - **Trend Detection**: Identify patterns
        - **Campaign Insights**: Per-campaign metrics
        - **Mistral AI**: AI-powered recommendations
        """)
    
    with col3:
        st.markdown("""
        ### 💰 Optimization
        - **Budget Allocation**: Optimize spending
        - **Constrained Optimization**: Set limits
        - **ROI Maximization**: Increase returns
        - **Reports**: Professional outputs
        - **Interactive Viz**: Charts & graphs
        """)
    
    st.markdown("---")
    
    st.markdown("---")
    
    # Data format requirements
    with st.expander("📋 CSV Data Format Requirements"):
        st.markdown("""
        ### Required Columns:
        - **campaign_id**: Unique campaign identifier (e.g., CMP-1001)
        - **campaign_name**: Descriptive campaign name (e.g., "Summer Sale 2024")
        - **channel**: Marketing channel (Email, Social Media, Paid Search, etc.)
        - **date**: Date in YYYY-MM-DD format (e.g., 2024-01-01)
        - **impressions**: Number of ad impressions
        - **clicks**: Number of clicks
        - **conversions**: Number of conversions
        - **cost**: Campaign cost in dollars
        - **revenue**: Revenue generated in dollars
        
        ### Optional Columns:
        - **conversion_rate**: Conversion rate (will be calculated if missing)
        - **roi**: Return on investment (will be calculated if missing)
        - **currency**: Currency code (e.g., USD)
        
        ### Example:
        ```
        campaign_id,campaign_name,channel,date,impressions,clicks,conversions,conversion_rate,cost,revenue,roi,currency
        CMP-1001,Newsletter Q1,Email,2024-01-01,10000,850,45,5.29,500.00,4500.00,800.00,USD
        CMP-1002,Facebook Campaign,Social Media,2024-01-01,100000,5000,120,2.40,1500.00,3600.00,140.00,USD
        ```
        """)
    
    # Current session status
    st.markdown("---")
    st.markdown("## 📊 Current Session Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.data_uploaded:
            st.success("✅ Data Loaded")
        else:
            st.warning("⏳ No Data")
    
    with col2:
        if st.session_state.analysis_complete:
            st.success("✅ Analysis Done")
        else:
            st.info("⏳ Not Analyzed")
    
    with col3:
        if st.session_state.optimization_complete:
            st.success("✅ Optimized")
        else:
            st.info("⏳ Not Optimized")
    
    with col4:
        records = len(st.session_state.marketing_data) if st.session_state.marketing_data is not None else 0
        st.metric("Records", records)

def show_data_input_page(pinecone_mgr, data_processor):
    """Data input page with CSV upload"""
    
    st.markdown("## 📥 Data Input")
    st.markdown("Upload your marketing campaign data or generate sample data for testing.")
    
    st.markdown("---")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📤 Upload CSV", "🎲 Generate Sample Data"])
    
    # Tab 1: CSV Upload
    with tab1:
        st.markdown("### Upload CSV File")
        
        # Download sample CSV template
        sample_df = pd.DataFrame({
            'campaign_id': ['CMP-1001', 'CMP-1002', 'CMP-1003'],
            'campaign_name': ['Newsletter Q1', 'Facebook Campaign', 'Google Search Ads'],
            'channel': ['Email', 'Social Media', 'Paid Search'],
            'date': ['2024-01-01', '2024-01-01', '2024-01-01'],
            'impressions': [10000, 100000, 150000],
            'clicks': [850, 5000, 7500],
            'conversions': [45, 120, 85],
            'conversion_rate': [5.29, 2.40, 1.13],
            'cost': [500.00, 1500.00, 2000.00],
            'revenue': [4500.00, 3600.00, 4250.00],
            'roi': [800.00, 140.00, 112.50],
            'currency': ['USD', 'USD', 'USD']
        })
        
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_df.to_csv(index=False),
            file_name="marketing_campaign_template.csv",
            mime="text/csv",
            help="Download a template CSV file with correct format"
        )
        
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with your marketing campaign data"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File loaded successfully! Found {len(df)} records")
                
                # Validate data
                st.markdown("### 🔍 Data Validation")
                
                required_cols = ['campaign_id', 'campaign_name', 'channel', 'date', 
                               'impressions', 'clicks', 'conversions', 'cost', 'revenue']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    st.success("✅ All required columns present")
                    
                    # Check for negative values
                    numeric_cols = ['impressions', 'clicks', 'conversions', 'cost', 'revenue']
                    has_negatives = False
                    for col in numeric_cols:
                        if (df[col] < 0).any():
                            st.error(f"❌ Negative values found in column: {col}")
                            has_negatives = True
                    
                    if not has_negatives:
                        st.success("✅ No negative values detected")
                        
                        # Convert date column
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # Calculate missing metrics
                        if 'roi' not in df.columns:
                            df['roi'] = ((df['revenue'] - df['cost']) / df['cost']) * 100
                        if 'conversion_rate' not in df.columns:
                            df['conversion_rate'] = (df['conversions'] / df['clicks']) * 100
                            df['conversion_rate'] = df['conversion_rate'].fillna(0)
                        
                        # Show data preview
                        st.markdown("### 📊 Data Preview")
                        st.dataframe(df.head(20), use_container_width=True)
                        
                        # Show data summary
                        st.markdown("### 📈 Data Summary")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Total Records", len(df))
                        with col2:
                            st.metric("Campaigns", df['campaign_id'].nunique())
                        with col3:
                            st.metric("Channels", len(df['channel'].unique()))
                        with col4:
                            st.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
                        with col5:
                            st.metric("Total Cost", format_currency(df['cost'].sum()))
                        
                        # Channel breakdown
                        st.markdown("#### Channel Breakdown")
                        channel_summary = df.groupby('channel').agg({
                            'cost': 'sum',
                            'revenue': 'sum',
                            'conversions': 'sum',
                            'clicks': 'sum',
                            'impressions': 'sum'
                        }).reset_index()
                        
                        channel_summary['ROI (%)'] = ((channel_summary['revenue'] - channel_summary['cost']) / channel_summary['cost'] * 100).round(2)
                        channel_summary['CTR (%)'] = (channel_summary['clicks'] / channel_summary['impressions'] * 100).round(2)
                        
                        st.dataframe(channel_summary, use_container_width=True)
                        
                        # Upload to Pinecone button
                        st.markdown("---")
                        st.markdown("### 📤 Upload to Pinecone")
                        
                        if st.button("🚀 Upload Data to Pinecone", type="primary", use_container_width=True):
                            with st.spinner("Uploading data to Pinecone vector database..."):
                                try:
                                    # Create index
                                    pinecone_mgr.create_index()
                                    
                                    # Upload data
                                    pinecone_mgr.upload_data(df)
                                    
                                    # Store in session state
                                    st.session_state.data_uploaded = True
                                    st.session_state.marketing_data = df
                                    
                                    st.success("✅ Data successfully uploaded to Pinecone!")
                                    st.balloons()
                                    
                                except Exception as e:
                                    st.error(f"❌ Error uploading data: {str(e)}")
                        
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
    
    # Tab 2: Generate Sample Data
    with tab2:
        st.markdown("### 🎲 Generate Sample Data")
        st.info("Generate realistic sample marketing campaign data for testing the system")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_days = st.slider("Number of days", 30, 180, 90)
        
        with col2:
            channels = st.multiselect(
                "Select channels",
                ['Email', 'Social Media', 'Paid Search', 'Organic Search', 'Display Ads'],
                default=['Email', 'Social Media', 'Paid Search', 'Organic Search', 'Display Ads']
            )
        
        if st.button("🎲 Generate Sample Data", use_container_width=True):
            with st.spinner("Generating sample data..."):
                df = generate_sample_data()
                
                # Filter by selected channels
                df = df[df['channel'].isin(channels)]
                
                # Filter by number of days
                df['date'] = pd.to_datetime(df['date'])
                max_date = df['date'].max()
                min_date = max_date - timedelta(days=num_days)
                df = df[df['date'] >= min_date]
                
                st.success(f"✅ Generated {len(df)} records across {df['campaign_id'].nunique()} campaigns!")
                
                # Show preview
                st.markdown("### 📊 Generated Data Preview")
                st.dataframe(df.head(20), use_container_width=True)
                
                # Show summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Campaigns", df['campaign_id'].nunique())
                with col2:
                    st.metric("Channels", df['channel'].nunique())
                with col3:
                    st.metric("Total Cost", format_currency(df['cost'].sum()))
                with col4:
                    st.metric("Total Revenue", format_currency(df['revenue'].sum()))
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Generated Data",
                    data=csv,
                    file_name=f"sample_marketing_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Upload to Pinecone
                st.markdown("---")
                if st.button("🚀 Upload to Pinecone", type="primary", use_container_width=True):
                    with st.spinner("Uploading to Pinecone..."):
                        try:
                            pinecone_mgr.create_index()
                            pinecone_mgr.upload_data(df)
                            
                            st.session_state.data_uploaded = True
                            st.session_state.marketing_data = df
                            
                            st.success("✅ Sample data uploaded successfully!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

def show_analysis_page(pinecone_mgr, mistral_agent):
    """AI analysis page"""
    
    st.markdown("## 📊 AI-Powered Analysis")
    
    if not st.session_state.data_uploaded:
        st.warning("⚠️ Please upload data first in the **Data Input** page")
        return
    
    st.markdown("Analyze your marketing performance with AI-powered insights from Mistral.")
    
    st.markdown("---")
    
    if st.button("🔍 Run AI Analysis", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing your marketing data with Mistral AI..."):
            try:
                # ADD THIS: Wait for index to be ready
                st.info("⏳ Waiting for Pinecone index to be ready...")
                time.sleep(3)  # Give Pinecone time to propagate
                
                # Retrieve data from Pinecone
                st.info("📥 Retrieving data from Pinecone...")
                channel_metrics = pinecone_mgr.get_aggregated_metrics()
                
                if not channel_metrics:
                    st.error("❌ No data found in Pinecone.")
                    
                    # FALLBACK: Use data from session state
                    st.warning("⚠️ Falling back to uploaded data from session...")
                    if st.session_state.marketing_data is not None:
                        from src.data_processor import DataProcessor
                        processor = DataProcessor()
                        channel_metrics = processor.aggregate_channel_metrics(st.session_state.marketing_data)
                        st.success("✓ Using data from session state")
                    else:
                        st.error("No data available. Please re-upload your data.")
                        return
                
                # Format context for Mistral
                context_data = format_context_for_mistral(channel_metrics)
                
                # Get analysis from Mistral
                st.info("🤖 Generating AI analysis with Mistral...")
                analysis = mistral_agent.analyze_marketing_data(context_data)
                
                # Store results in session state
                st.session_state.channel_metrics = channel_metrics
                st.session_state.analysis_result = analysis
                st.session_state.analysis_complete = True
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.channel_metrics:
        
        st.markdown("---")
        st.markdown("## 📈 Analysis Results")
        
        channel_metrics = st.session_state.channel_metrics
        
        # Overall metrics
        st.markdown("### 🎯 Overall Performance")
        
        total_cost = sum(m['cost'] for m in channel_metrics.values())
        total_revenue = sum(m['revenue'] for m in channel_metrics.values())
        total_conversions = sum(m['conversions'] for m in channel_metrics.values())
        total_clicks = sum(m['clicks'] for m in channel_metrics.values())
        total_impressions = sum(m['impressions'] for m in channel_metrics.values())
        overall_roi = ((total_revenue - total_cost) / total_cost * 100)
        overall_ctr = (total_clicks / total_impressions * 100)
        overall_conv_rate = (total_conversions / total_clicks * 100)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Cost", format_currency(total_cost))
        with col2:
            st.metric("Total Revenue", format_currency(total_revenue))
        with col3:
            st.metric("Conversions", f"{total_conversions:,}")
        with col4:
            st.metric("ROI", format_percentage(overall_roi))
        with col5:
            st.metric("Conv. Rate", format_percentage(overall_conv_rate))
        
        st.markdown("---")
        
        # Channel performance table
        st.markdown("### 📊 Channel Performance Metrics")
        
        metrics_df = pd.DataFrame.from_dict(channel_metrics, orient='index')
        metrics_df = metrics_df.reset_index().rename(columns={'index': 'channel'})
        metrics_df = metrics_df.sort_values('roi', ascending=False)
        
        # Format for display
        display_df = metrics_df.copy()
        display_df['cost'] = display_df['cost'].apply(format_currency)
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['roi'] = display_df['roi'].apply(format_percentage)
        display_df['cpa'] = display_df['cpa'].apply(format_currency)
        display_df['ctr'] = display_df['ctr'].apply(format_percentage)
        display_df['conversion_rate'] = display_df['conversion_rate'].apply(format_percentage)
        display_df['conversions'] = display_df['conversions'].apply(lambda x: f"{x:,}")
        display_df['clicks'] = display_df['clicks'].apply(lambda x: f"{x:,}")
        display_df['impressions'] = display_df['impressions'].apply(lambda x: f"{x:,}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📈 Performance Visualizations")
        
        # ROI by Channel
        fig_roi = px.bar(
            metrics_df,
            x='channel',
            y='roi',
            title="ROI by Channel (%)",
            labels={'roi': 'ROI (%)', 'channel': 'Channel'},
            color='roi',
            color_continuous_scale='RdYlGn'
        )
        fig_roi.update_layout(showlegend=False)
        st.plotly_chart(fig_roi, use_container_width=True)
        
        # Cost vs Revenue
        fig_cost_revenue = go.Figure()
        fig_cost_revenue.add_trace(go.Bar(
            x=metrics_df['channel'],
            y=metrics_df['cost'],
            name='Cost',
            marker_color='indianred'
        ))
        fig_cost_revenue.add_trace(go.Bar(
            x=metrics_df['channel'],
            y=metrics_df['revenue'],
            name='Revenue',
            marker_color='lightseagreen'
        ))
        fig_cost_revenue.update_layout(
            title="Cost vs Revenue by Channel",
            xaxis_title="Channel",
            yaxis_title="Amount ($)",
            barmode='group'
        )
        st.plotly_chart(fig_cost_revenue, use_container_width=True)
        
        # Conversion funnel metrics
        col1, col2 = st.columns(2)
        
        with col1:
            fig_ctr = px.bar(
                metrics_df,
                x='channel',
                y='ctr',
                title="Click-Through Rate (CTR) by Channel",
                labels={'ctr': 'CTR (%)', 'channel': 'Channel'},
                color='ctr',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_ctr, use_container_width=True)
        
        with col2:
            fig_conv_rate = px.bar(
                metrics_df,
                x='channel',
                y='conversion_rate',
                title="Conversion Rate by Channel",
                labels={'conversion_rate': 'Conversion Rate (%)', 'channel': 'Channel'},
                color='conversion_rate',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_conv_rate, use_container_width=True)
        
        # CPA comparison
        fig_cpa = px.bar(
            metrics_df,
            x='channel',
            y='cpa',
            title="Cost Per Acquisition (CPA) by Channel",
            labels={'cpa': 'CPA ($)', 'channel': 'Channel'},
            color='cpa',
            color_continuous_scale='Reds_r'
        )
        st.plotly_chart(fig_cpa, use_container_width=True)
        
        # Conversions distribution
        fig_conversions = px.pie(
            metrics_df,
            values='conversions',
            names='channel',
            title="Conversions Distribution by Channel"
        )
        st.plotly_chart(fig_conversions, use_container_width=True)
        
        st.markdown("---")
        
        # AI Analysis from Mistral
        st.markdown("### 🤖 AI-Generated Insights (Mistral)")
        
        if st.session_state.analysis_result:
            st.markdown(st.session_state.analysis_result)

def show_optimization_page(pinecone_mgr, mistral_agent):
    """Budget optimization page"""
    
    st.markdown("## 💰 Budget Optimization")
    
    if not st.session_state.analysis_complete:
        st.warning("⚠️ Please run the analysis first in the **AI Analysis** page")
        return
    
    st.markdown("Optimize your marketing budget allocation to maximize ROI and conversions.")
    
    st.markdown("---")
    
    # Budget input
    st.markdown("### 💵 Budget Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_budget = st.number_input(
            "Total Marketing Budget ($)",
            min_value=1000.0,
            value=50000.0,
            step=1000.0,
            help="Enter your total available marketing budget"
        )
    
    with col2:
        optimization_goal = st.selectbox(
            "Optimization Goal",
            ["Maximize Conversions", "Maximize Revenue", "Maximize ROI"],
            help="Select what you want to optimize for"
        )
    
    # Constraints
    st.markdown("### ⚙️ Constraints")
    
    # Calculate safe defaults based on number of channels
    if st.session_state.channel_metrics:
        n_channels = len(st.session_state.channel_metrics)
        safe_min = min(5.0, 95.0 / n_channels)  # Ensure feasibility
        safe_max = max(50.0, 100.0 / n_channels + 20)  # Ensure flexibility
    else:
        safe_min = 5.0
        safe_max = 50.0
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_allocation = st.slider(
            "Minimum Allocation per Channel (%)",
            0.0, 20.0, float(safe_min),
            step=1.0,
            help=f"Minimum percentage of budget each channel must receive (Recommended: {safe_min:.1f}%)"
        )
    
    with col2:
        max_allocation = st.slider(
            "Maximum Allocation per Channel (%)",
            20.0, 100.0, float(safe_max),
            step=5.0,
            help=f"Maximum percentage of budget each channel can receive (Recommended: {safe_max:.1f}%)"
        )
    
    # Validation warning
    if st.session_state.channel_metrics:
        n_channels = len(st.session_state.channel_metrics)
        min_total = n_channels * min_allocation
        
        if min_total > 100:
            st.error(f"⚠️ **Constraint Error**: Minimum allocation ({min_allocation}% × {n_channels} channels = {min_total}%) exceeds 100%!")
            st.info(f"💡 Reduce minimum to {100.0/n_channels:.1f}% or less")
        elif min_total > 95:
            st.warning(f"⚠️ **Warning**: Very tight constraints ({min_total:.1f}% minimum). Optimization may be limited.")
    
    st.markdown("---")
    
    if st.button("🚀 Optimize Budget Allocation", type="primary", use_container_width=True):
        with st.spinner("🔄 Running optimization algorithm..."):
            try:
                channel_metrics = st.session_state.channel_metrics
                
                # Prepare data for optimizer
                metrics_df = pd.DataFrame.from_dict(channel_metrics, orient='index').reset_index()
                metrics_df = metrics_df.rename(columns={'index': 'channel'})
                
                # Run optimization
                optimizer = BudgetOptimizer(metrics_df, total_budget)
                optimal_allocation = optimizer.optimize(
                    min_allocation_pct=min_allocation/100,
                    max_allocation_pct=max_allocation/100
                )
                
                improvement_summary = optimizer.improvement_summary
                
                # Store results
                st.session_state.optimization_result = {
                    'allocation': optimal_allocation,
                    'improvement': improvement_summary
                }
                st.session_state.optimization_complete = True
                
                st.success("✅ Optimization complete!")
                
                # Get AI recommendations
                st.info("🤖 Generating AI recommendations...")
                
                current_allocation = {
                    row['channel']: {
                        'amount': row['cost'],
                        'percentage': (row['cost'] / total_budget) * 100
                    }
                    for _, row in metrics_df.iterrows()
                }
                
                ai_recommendations = mistral_agent.generate_optimization_recommendations(
                    channel_metrics,
                    total_budget,
                    current_allocation
                )
                
                st.session_state.optimization_result['ai_recommendations'] = ai_recommendations
                
            except Exception as e:
                st.error(f"❌ Optimization error: {str(e)}")
                return
    
    # Display optimization results
    if st.session_state.optimization_complete and st.session_state.optimization_result:
        
        st.markdown("---")
        st.markdown("## 📊 Optimization Results")
        
        optimal_allocation = st.session_state.optimization_result['allocation']
        improvement_summary = st.session_state.optimization_result['improvement']
        
        # Improvement summary
        st.markdown("### 📈 Expected Improvements")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Conversion Improvement",
                f"+{improvement_summary['conversion_improvement']:.0f}",
                f"+{improvement_summary['conversion_improvement_pct']:.1f}%"
            )
        
        with col2:
            current_roi = improvement_summary['current_roi']
            optimal_roi = improvement_summary['optimal_roi']
            st.metric(
                "ROI Improvement",
                format_percentage(optimal_roi),
                f"+{optimal_roi - current_roi:.2f}%"
            )
        
        with col3:
            st.metric(
                "Total Budget",
                format_currency(total_budget)
            )
        
        st.markdown("---")
        
        # Allocation comparison table
        st.markdown("### 💵 Budget Allocation Recommendations")
        
        display_allocation = optimal_allocation.copy()
        display_allocation['current_cost'] = display_allocation['current_cost'].apply(format_currency)
        display_allocation['optimal_budget'] = display_allocation['optimal_budget'].apply(format_currency)
        display_allocation['budget_change'] = display_allocation['budget_change'].apply(format_currency)
        display_allocation['budget_change_pct'] = display_allocation['budget_change_pct'].apply(lambda x: f"{x:.1f}%")
        display_allocation['expected_roi'] = display_allocation['expected_roi'].apply(format_percentage)
        
        st.dataframe(display_allocation, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📊 Allocation Visualizations")
        
        # Current vs Optimal pie charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_current = px.pie(
                optimal_allocation,
                values='current_cost',
                names='channel',
                title="Current Budget Allocation"
            )
            st.plotly_chart(fig_current, use_container_width=True)
        
        with col2:
            fig_optimal = px.pie(
                optimal_allocation,
                values='optimal_budget',
                names='channel',
                title="Optimized Budget Allocation"
            )
            st.plotly_chart(fig_optimal, use_container_width=True)
        
        # Budget change waterfall
        fig_waterfall = go.Figure(go.Waterfall(
            name="Budget Changes",
            orientation="v",
            x=optimal_allocation['channel'],
            y=optimal_allocation['budget_change'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(
            title="Budget Changes by Channel ($)",
            showlegend=False
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.markdown("---")
        
        # AI Recommendations
        if 'ai_recommendations' in st.session_state.optimization_result:
            st.markdown("### 🤖 AI-Generated Recommendations (Mistral)")
            st.markdown(st.session_state.optimization_result['ai_recommendations'])

def show_report_page(pinecone_mgr, mistral_agent):
    """Report generation page"""
    
    st.markdown("## 📄 Generate Comprehensive Report")
    
    if not st.session_state.optimization_complete:
        st.warning("⚠️ Please complete the optimization first in the **Budget Optimization** page")
        return
    
    st.markdown("Generate a comprehensive text report with all analysis and optimization results.")
    
    st.markdown("---")
    
    # Report configuration
    st.markdown("### ⚙️ Report Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input("Company Name", "Your Company")
        report_title = st.text_input("Report Title", "Marketing ROI Analysis & Budget Optimization")
    
    with col2:
        report_date = st.date_input("Report Date", datetime.now())
        report_format = st.selectbox("Report Format", ["📄 Text File (.txt)", "📊 Markdown (.md)", "📋 CSV Summary"])
    
    st.markdown("---")
    
    if st.button("📄 Generate Report", type="primary", use_container_width=True):
        with st.spinner("📝 Generating comprehensive report..."):
            try:
                channel_metrics = st.session_state.channel_metrics
                optimal_allocation = st.session_state.optimization_result['allocation']
                improvement_summary = st.session_state.optimization_result['improvement']
                
                # Calculate overall metrics
                total_cost = sum(m['cost'] for m in channel_metrics.values())
                total_revenue = sum(m['revenue'] for m in channel_metrics.values())
                total_conversions = sum(m['conversions'] for m in channel_metrics.values())
                overall_roi = ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
                
                # Generate report content based on format
                if "Text" in report_format:
                    report_content = f"""
{'='*80}
{report_title.upper()}
{'='*80}

Company: {company_name}
Report Date: {report_date.strftime('%B %d, %Y')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Powered by: Pinecone RAG + Mistral AI

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

"""
                    # Get executive summary from Mistral
                    if st.session_state.analysis_result and 'ai_recommendations' in st.session_state.optimization_result:
                        try:
                            exec_summary = mistral_agent.generate_executive_summary(
                                st.session_state.analysis_result,
                                st.session_state.optimization_result['ai_recommendations']
                            )
                            report_content += exec_summary + "\n\n"
                        except:
                            report_content += "Executive summary unavailable.\n\n"
                    
                    report_content += f"""
{'='*80}
OVERALL PERFORMANCE METRICS
{'='*80}

Total Cost:          ${total_cost:,.2f}
Total Revenue:       ${total_revenue:,.2f}
Total Conversions:   {total_conversions:,}
Overall ROI:         {overall_roi:.2f}%
Number of Channels:  {len(channel_metrics)}

{'='*80}
CHANNEL PERFORMANCE BREAKDOWN
{'='*80}

"""
                    # Channel metrics table
                    for channel, metrics in sorted(channel_metrics.items(), key=lambda x: x[1]['roi'], reverse=True):
                        report_content += f"""
Channel: {channel}
{'-'*60}
Cost:              ${metrics['cost']:,.2f}
Revenue:           ${metrics['revenue']:,.2f}
Conversions:       {metrics['conversions']:,}
Clicks:            {metrics['clicks']:,}
Impressions:       {metrics['impressions']:,}
ROI:               {metrics['roi']:.2f}%
CPA:               ${metrics['cpa']:.2f}
CTR:               {metrics['ctr']:.2f}%
Conversion Rate:   {metrics['conversion_rate']:.2f}%
Records:           {metrics['records']}

"""
                    
                    report_content += f"""
{'='*80}
BUDGET OPTIMIZATION RESULTS
{'='*80}

Expected Improvements:
- Conversion Improvement:    +{improvement_summary['conversion_improvement']:.0f} ({improvement_summary['conversion_improvement_pct']:.1f}%)
- Current ROI:               {improvement_summary['current_roi']:.2f}%
- Optimized ROI:             {improvement_summary['optimal_roi']:.2f}%
- ROI Improvement:           +{improvement_summary['roi_improvement']:.2f}%

{'='*80}
RECOMMENDED BUDGET ALLOCATION
{'='*80}

"""
                    # Budget allocation table
                    for _, row in optimal_allocation.iterrows():
                        report_content += f"""
Channel: {row['channel']}
{'-'*60}
Current Cost:      ${row['current_cost']:,.2f}
Optimal Budget:    ${row['optimal_budget']:,.2f}
Budget Change:     ${row['budget_change']:,.2f} ({row['budget_change_pct']:.1f}%)
Expected ROI:      {row['expected_roi']:.2f}%

"""
                    
                    # AI Recommendations
                    if 'ai_recommendations' in st.session_state.optimization_result:
                        report_content += f"""
{'='*80}
AI-GENERATED RECOMMENDATIONS
{'='*80}

{st.session_state.optimization_result['ai_recommendations']}

"""
                    
                    report_content += f"""
{'='*80}
END OF REPORT
{'='*80}
"""
                    
                    file_extension = "txt"
                    mime_type = "text/plain"
                
                elif "Markdown" in report_format:
                    # Markdown format
                    report_content = f"""# {report_title}

**Company:** {company_name}  
**Report Date:** {report_date.strftime('%B %d, %Y')}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Powered by:** Pinecone RAG + Mistral AI

---

## Executive Summary

"""
                    if st.session_state.analysis_result and 'ai_recommendations' in st.session_state.optimization_result:
                        try:
                            exec_summary = mistral_agent.generate_executive_summary(
                                st.session_state.analysis_result,
                                st.session_state.optimization_result['ai_recommendations']
                            )
                            report_content += exec_summary + "\n\n"
                        except:
                            pass
                    
                    report_content += f"""
---

## Overall Performance Metrics

| Metric | Value |
|--------|-------|
| Total Cost | ${total_cost:,.2f} |
| Total Revenue | ${total_revenue:,.2f} |
| Total Conversions | {total_conversions:,} |
| Overall ROI | {overall_roi:.2f}% |
| Number of Channels | {len(channel_metrics)} |

---

## Channel Performance Breakdown

"""
                    for channel, metrics in sorted(channel_metrics.items(), key=lambda x: x[1]['roi'], reverse=True):
                        report_content += f"""
### {channel}

| Metric | Value |
|--------|-------|
| Cost | ${metrics['cost']:,.2f} |
| Revenue | ${metrics['revenue']:,.2f} |
| ROI | {metrics['roi']:.2f}% |
| CPA | ${metrics['cpa']:.2f} |
| Conversions | {metrics['conversions']:,} |
| CTR | {metrics['ctr']:.2f}% |
| Conversion Rate | {metrics['conversion_rate']:.2f}% |

"""
                    
                    report_content += f"""
---

## Budget Optimization Results

### Expected Improvements

- **Conversion Improvement:** +{improvement_summary['conversion_improvement']:.0f} ({improvement_summary['conversion_improvement_pct']:.1f}%)
- **Current ROI:** {improvement_summary['current_roi']:.2f}%
- **Optimized ROI:** {improvement_summary['optimal_roi']:.2f}%
- **ROI Improvement:** +{improvement_summary['roi_improvement']:.2f}%

### Recommended Budget Allocation

| Channel | Current Cost | Optimal Budget | Change | Expected ROI |
|---------|-------------|----------------|--------|--------------|
"""
                    for _, row in optimal_allocation.iterrows():
                        report_content += f"| {row['channel']} | ${row['current_cost']:,.2f} | ${row['optimal_budget']:,.2f} | {row['budget_change_pct']:.1f}% | {row['expected_roi']:.2f}% |\n"
                    
                    if 'ai_recommendations' in st.session_state.optimization_result:
                        report_content += f"""
---

## AI-Generated Recommendations

{st.session_state.optimization_result['ai_recommendations']}
"""
                    
                    file_extension = "md"
                    mime_type = "text/markdown"
                
                else:  # CSV Summary
                    # Create CSV with key metrics
                    csv_data = []
                    csv_data.append(["Marketing ROI Analysis & Budget Optimization Report"])
                    csv_data.append([f"Company: {company_name}"])
                    csv_data.append([f"Report Date: {report_date.strftime('%B %d, %Y')}"])
                    csv_data.append([])
                    csv_data.append(["OVERALL METRICS"])
                    csv_data.append(["Metric", "Value"])
                    csv_data.append(["Total Cost", f"${total_cost:,.2f}"])
                    csv_data.append(["Total Revenue", f"${total_revenue:,.2f}"])
                    csv_data.append(["Total Conversions", f"{total_conversions:,}"])
                    csv_data.append(["Overall ROI", f"{overall_roi:.2f}%"])
                    csv_data.append([])
                    csv_data.append(["CHANNEL PERFORMANCE"])
                    csv_data.append(["Channel", "Cost", "Revenue", "Conversions", "ROI", "CPA", "CTR", "Conv Rate"])
                    
                    for channel, metrics in channel_metrics.items():
                        csv_data.append([
                            channel,
                            f"${metrics['cost']:,.2f}",
                            f"${metrics['revenue']:,.2f}",
                            f"{metrics['conversions']:,}",
                            f"{metrics['roi']:.2f}%",
                            f"${metrics['cpa']:.2f}",
                            f"{metrics['ctr']:.2f}%",
                            f"{metrics['conversion_rate']:.2f}%"
                        ])
                    
                    csv_data.append([])
                    csv_data.append(["OPTIMIZED BUDGET ALLOCATION"])
                    csv_data.append(["Channel", "Current Cost", "Optimal Budget", "Change %", "Expected ROI"])
                    
                    for _, row in optimal_allocation.iterrows():
                        csv_data.append([
                            row['channel'],
                            f"${row['current_cost']:,.2f}",
                            f"${row['optimal_budget']:,.2f}",
                            f"{row['budget_change_pct']:.1f}%",
                            f"{row['expected_roi']:.2f}%"
                        ])
                    
                    import io
                    import csv
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerows(csv_data)
                    report_content = output.getvalue()
                    
                    file_extension = "csv"
                    mime_type = "text/csv"
                
                # Save and provide download
                report_filename = f"marketing_roi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
                
                st.success(f"✅ Report generated successfully!")
                
                # Provide download button
                st.download_button(
                    label=f"📥 Download {report_format}",
                    data=report_content,
                    file_name=report_filename,
                    mime=mime_type,
                    use_container_width=True
                )
                
                # Show preview
                with st.expander("📄 Report Preview"):
                    if file_extension == "txt":
                        st.text(report_content)
                    elif file_extension == "md":
                        st.markdown(report_content)
                    else:
                        st.text(report_content)
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
    
    st.markdown("---")
    
    # Export data options
    st.markdown("### 📤 Export Raw Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.marketing_data is not None:
            csv = st.session_state.marketing_data.to_csv(index=False)
            st.download_button(
                label="📥 Export Raw Data (CSV)",
                data=csv,
                file_name=f"marketing_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.session_state.channel_metrics:
            metrics_df = pd.DataFrame.from_dict(st.session_state.channel_metrics, orient='index')
            metrics_csv = metrics_df.to_csv()
            st.download_button(
                label="📥 Export Metrics (CSV)",
                data=metrics_csv,
                file_name=f"channel_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
def show_qa_page(pinecone_mgr, mistral_agent):
    """Interactive Q&A page"""
    
    st.markdown("## ❓ Ask Questions About Your Data")
    
    if not st.session_state.data_uploaded:
        st.warning("⚠️ Please upload data first in the **Data Input** page")
        return
    
    st.markdown("Ask specific questions about your marketing campaigns and get AI-powered answers instantly.")
    
    st.markdown("---")
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        st.markdown("""
        - Which channel has the highest ROI?
        - What's my average cost per acquisition?
        - Which campaigns are underperforming?
        - How much should I invest in Social Media?
        - What's the best performing channel this month?
        - How can I improve my email marketing ROI?
        - Which channel has the best conversion rate?
        - What's driving my conversions?
        - Compare cost per click across channels
        - Which campaigns should I scale up?
        """)
    
    # Chat interface
    st.markdown("### 💬 Chat with Your Data")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🙋 You:** {question}")
            st.markdown(f"**🤖 AI:** {answer}")
            st.markdown("---")
    
    # Question input
    question = st.text_input("Ask a question:", placeholder="e.g., Which channel should I invest more in?")
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        if st.button("🔍 Ask Question", type="primary", use_container_width=True):
            if question:
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Query Pinecone for relevant context
                        results = pinecone_mgr.query_data(question, top_k=30)
                        
                        # Format context
                        qa_context = "Relevant Marketing Campaign Data:\n\n"
                        for match in results['matches'][:10]:
                            metadata = match['metadata']
                            qa_context += f"{metadata.get('text', '')}\n\n"
                        
                        # Get answer from Mistral
                        answer = mistral_agent.answer_specific_question(qa_context, question)
                        
                        # Add to chat history
                        st.session_state.chat_history.append((question, answer))
                        
                        # Rerun to display new message
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please enter a question")
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Run the app
if __name__ == "__main__":
    main()
