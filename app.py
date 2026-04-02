import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import os
import requests
import urllib.parse
import re
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="EV Adoption Analysis Dashboard",
    page_icon="bar-chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Custom Theme & Styling ---
# A modern dark background with professional, muted but legible accent tones
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0d1117; /* GitHub Dark dimension */
        color: #c9d1d9;
    }
    
    /* Headers & Text */
    h1, h2, h3, h4, h5, h6 {
        color: #58a6ff !important; /* Soft Blue */
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric Cards styled */
    div[data-testid="stMetricValue"] {
        color: #3fb950 !important; /* Emerald/Green */
        font-size: 2rem !important;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 1.1rem !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #8b949e !important; 
    }
</style>
""", unsafe_allow_html=True)

# Define robust professional color palette for charts
COLORS = {
    'primary': '#58a6ff',   # blue
    'secondary': '#3fb950', # green
    'tertiary': '#bc8cff',  # violet
    'quaternary': '#d29922', # amber/bronze
    'quinary': '#ff7b72',   # coral
    'senary': '#2f81f7',    # deeper blue
    'slate': '#8b949e'      # grey
}
PLOTLY_THEME = 'plotly_dark'

# --- Data Loading ---
@st.cache_data
def load_data():
    file_path = "cleaned_data.csv"
    if not os.path.exists(file_path):
        st.error(f"Dataset not found at {file_path}. Please ensure it is present in the same directory.")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Preprocessing validation
    # Cast date properly
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Handle optional missing values
    df = df.dropna()
    
    return df

with st.spinner("Loading and preprocessing data..."):
    data = load_data()

if data.empty:
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Year Filter
unique_years = sorted(data['year'].unique())
selected_year = st.sidebar.multiselect(
    "Select Year",
    options=unique_years,
    default=unique_years
)

# State Filter
unique_states = sorted(data['state'].unique())
selected_state = st.sidebar.multiselect(
    "Select State",
    options=unique_states,
    default=unique_states[:5] if len(unique_states) > 5 else unique_states
)

# Vehicle Category Filter
unique_categories = sorted(data['vehicle_category'].unique())
selected_category = st.sidebar.multiselect(
    "Select Vehicle Category",
    options=unique_categories,
    default=unique_categories
)

# Apply Filters
filtered_data = data[
    (data['year'].isin(selected_year)) &
    (data['state'].isin(selected_state)) &
    (data['vehicle_category'].isin(selected_category))
]

if filtered_data.empty:
    st.warning("No data matches the selected filters. Please adjust your criteria.")
    st.stop()

# --- Dashboard Overview ---
st.title("EV Adoption Patterns & Trends")
st.markdown("Analyze electric vehicle adoption, compare across regions, and discover underlying market segments.")

st.header("1. Dashboard Overview", divider="blue")

col1, col2, col3 = st.columns(3)

total_ev_sales = filtered_data['electric_vehicles_sold'].sum()
avg_ev_sales = filtered_data['electric_vehicles_sold'].mean()
avg_market_share = filtered_data['ev_market_share'].mean() * 100 # percentage

col1.metric("Total EV Sales", f"{total_ev_sales:,.0f}")
col2.metric("Average EV Sales per Record", f"{avg_ev_sales:,.1f}")
col3.metric("Average EV Market Share", f"{avg_market_share:.2f}%")

st.subheader("Data Preview")
st.dataframe(filtered_data.head(10), use_container_width=True)


# --- Exploratory Data Analysis (EDA) ---
st.header("2. Exploratory Data Analysis", divider="blue")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Year-wise EV Adoption Trend")
    # Group by year and sum EV sales
    yearly_trend = filtered_data.groupby('year')['electric_vehicles_sold'].sum().reset_index()
    fig_year = px.line(
        yearly_trend, x='year', y='electric_vehicles_sold', 
        markers=True, 
        title="Total EV Sales Over Years",
        template=PLOTLY_THEME,
        color_discrete_sequence=[COLORS['primary']]
    )
    st.plotly_chart(fig_year, use_container_width=True)

with col_right:
    st.subheader("Month-wise Sales Trend")
    monthly_trend = filtered_data.groupby(['year', 'month'])['electric_vehicles_sold'].sum().reset_index()
    # Create a nice datetime-like axis for plotting
    monthly_trend['period'] = pd.to_datetime(
        monthly_trend['year'].astype(str) + '-' + monthly_trend['month'].astype(str) + '-01'
    )
    fig_month = px.line(
        monthly_trend, x='period', y='electric_vehicles_sold',
        title="Monthly EV Sales Trend",
        template=PLOTLY_THEME,
        color_discrete_sequence=[COLORS['quaternary']]
    )
    st.plotly_chart(fig_month, use_container_width=True)

st.subheader("State-wise EV Sales Comparison")
state_sales = filtered_data.groupby('state')['electric_vehicles_sold'].sum().reset_index()
state_sales = state_sales.sort_values(by='electric_vehicles_sold', ascending=False)
fig_state = px.bar(
    state_sales, x='state', y='electric_vehicles_sold',
    title="EV Sales by State",
    template=PLOTLY_THEME,
    color='electric_vehicles_sold',
    color_continuous_scale='Blues'
)
# Ensure x-axis shows categorical states if they are numeric
fig_state.update_xaxes(type='category')
st.plotly_chart(fig_state, use_container_width=True)

# Geographic Map Mapping (Indian State Codes to Approximate Lat/Lon)
# state column holds integers representing states
STATE_COORDS = {
    1: {'name': 'Jammu & Kashmir', 'lat': 33.7782, 'lon': 76.5762},
    2: {'name': 'Himachal Pradesh', 'lat': 31.1048, 'lon': 77.1665},
    3: {'name': 'Punjab', 'lat': 31.1471, 'lon': 75.3412},
    4: {'name': 'Chandigarh', 'lat': 30.7333, 'lon': 76.7794},
    5: {'name': 'Uttarakhand', 'lat': 30.0668, 'lon': 79.0193},
    6: {'name': 'Haryana', 'lat': 29.0588, 'lon': 76.0856},
    7: {'name': 'Delhi', 'lat': 28.7041, 'lon': 77.1025},
    8: {'name': 'Rajasthan', 'lat': 27.0238, 'lon': 74.2179},
    9: {'name': 'Uttar Pradesh', 'lat': 26.8467, 'lon': 80.9462},
    10: {'name': 'Bihar', 'lat': 25.0961, 'lon': 85.3131},
    11: {'name': 'Sikkim', 'lat': 27.5330, 'lon': 88.5122},
    12: {'name': 'Arunachal Pradesh', 'lat': 28.2180, 'lon': 94.7278},
    13: {'name': 'Nagaland', 'lat': 26.1584, 'lon': 94.5624},
    14: {'name': 'Manipur', 'lat': 24.6637, 'lon': 93.9063},
    15: {'name': 'Mizoram', 'lat': 23.1645, 'lon': 92.9376},
    16: {'name': 'Tripura', 'lat': 23.9408, 'lon': 91.9882},
    17: {'name': 'Meghalaya', 'lat': 25.4670, 'lon': 91.3662},
    18: {'name': 'Assam', 'lat': 26.2006, 'lon': 92.9376},
    19: {'name': 'West Bengal', 'lat': 22.9868, 'lon': 87.8550},
    20: {'name': 'Jharkhand', 'lat': 23.6102, 'lon': 85.2799},
    21: {'name': 'Odisha', 'lat': 20.9517, 'lon': 85.0985},
    22: {'name': 'Chhattisgarh', 'lat': 21.2787, 'lon': 81.8661},
    23: {'name': 'Madhya Pradesh', 'lat': 22.9734, 'lon': 78.6569},
    24: {'name': 'Gujarat', 'lat': 22.2587, 'lon': 71.1924},
    26: {'name': 'Dadra & Nagar Haveli', 'lat': 20.1809, 'lon': 73.0169},
    27: {'name': 'Maharashtra', 'lat': 19.7515, 'lon': 75.7139},
    28: {'name': 'Andhra Pradesh', 'lat': 15.9129, 'lon': 79.7400},
    29: {'name': 'Karnataka', 'lat': 15.3173, 'lon': 75.7139},
    30: {'name': 'Goa', 'lat': 15.2993, 'lon': 74.1240},
    31: {'name': 'Lakshadweep', 'lat': 10.5667, 'lon': 72.6417},
    32: {'name': 'Kerala', 'lat': 10.8505, 'lon': 76.2711},
    33: {'name': 'Tamil Nadu', 'lat': 11.1271, 'lon': 78.6569},
    34: {'name': 'Puducherry', 'lat': 11.9416, 'lon': 79.8083},
    35: {'name': 'Andaman & Nicobar', 'lat': 11.7401, 'lon': 92.6586},
    36: {'name': 'Telangana', 'lat': 18.1124, 'lon': 79.0193}
}

st.subheader("Geographic EV Map (State Representation)")
map_data = []
for idx, row in state_sales.iterrows():
    s_code = row['state']
    if s_code in STATE_COORDS:
        info = STATE_COORDS[s_code]
        map_data.append({
            'State Code': s_code,
            'State Name': info['name'],
            'Lat': info['lat'],
            'Lon': info['lon'],
            'EV Sales': row['electric_vehicles_sold']
        })

if map_data:
    df_map = pd.DataFrame(map_data)
    fig_map = px.scatter_mapbox(
        df_map, lat="Lat", lon="Lon", hover_name="State Name", hover_data=["EV Sales"],
        color="EV Sales", size="EV Sales",
        color_continuous_scale=px.colors.sequential.Plotly3,
        size_max=40, zoom=3, mapbox_style="carto-darkmatter",
        title="EV Distribution Map"
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No geospatial mapping available for current state filters.")


col_left_2, col_right_2 = st.columns(2)

with col_left_2:
    st.subheader("Vehicle Category Distribution")
    cat_dist = filtered_data.groupby('vehicle_category')['electric_vehicles_sold'].sum().reset_index()
    fig_cat = px.pie(
        cat_dist, names='vehicle_category', values='electric_vehicles_sold',
        title="EV Sales by Vehicle Category",
        template=PLOTLY_THEME,
        color_discrete_sequence=[COLORS['tertiary'], COLORS['secondary'], COLORS['quinary']]
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with col_right_2:
    st.subheader("EV Sales Distribution")
    fig_hist = px.histogram(
        filtered_data, x='electric_vehicles_sold', 
        nbins=50,
        title="Distribution of EV Sales Volumes",
        template=PLOTLY_THEME,
        color_discrete_sequence=[COLORS['primary']]
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# --- Statistical Analysis ---
st.header("3. Statistical Analysis", divider="blue")

st.subheader("Summary Statistics")
# Show mean, median, variance etc but in a clean way
stats_cols = ['electric_vehicles_sold', 'total_vehicles_sold', 'ev_market_share']
summary_stats = filtered_data[stats_cols].describe().T
summary_stats['variance'] = filtered_data[stats_cols].var()
st.dataframe(summary_stats, use_container_width=True)

st.subheader("Feature Correlation")
# Calculate correlation matrix
corr_matrix = filtered_data[['electric_vehicles_sold', 'total_vehicles_sold', 'ev_market_share', 'month', 'year']].corr()

fig_corr = px.imshow(
    corr_matrix, 
    text_auto=True, 
    aspect="auto",
    color_continuous_scale='Blues',
    title="Correlation Matrix"
)
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("EV Sales vs Total Vehicle Sales")
fig_scatter_sales = px.scatter(
    filtered_data, x='total_vehicles_sold', y='electric_vehicles_sold',
    opacity=0.6,
    color='ev_market_share',
    color_continuous_scale='Viridis',
    title="Relationship Between Total Sales and EV Sales",
    template=PLOTLY_THEME
)
st.plotly_chart(fig_scatter_sales, use_container_width=True)


# --- Data Mining Technique (Core): Clustering ---
st.header("4. Segmentation (K-Means Clustering)", divider="blue")

st.markdown("""
We use **K-Means Clustering** on `electric_vehicles_sold`, `total_vehicles_sold`, and `ev_market_share` to dynamically segment the selected data into **High**, **Medium**, and **Low** EV adoption regions/periods.
""")

# Prepare data for clustering
cluster_features = ['electric_vehicles_sold', 'total_vehicles_sold', 'ev_market_share']
cluster_data = filtered_data[cluster_features].dropna().copy()

if len(cluster_data) >= 3:
    # Normalize data for K-Means
    cluster_data_norm = (cluster_data - cluster_data.mean()) / cluster_data.std()
    cluster_data_norm = cluster_data_norm.fillna(0) # In case std is 0
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_data_norm)
    
    cluster_data['Cluster_ID'] = cluster_labels
    
    # Map cluster IDs to Meaningful Labels based on average EV sales
    # Find which cluster has highest/lowest avg EV sales
    cluster_means = cluster_data.groupby('Cluster_ID')['electric_vehicles_sold'].mean().sort_values()
    
    # 0 -> low config, 1-> medium config, 2-> high config (sorted)
    mapping_order = list(cluster_means.index)
    label_mapping = {
        mapping_order[0]: "Low Adoption",
        mapping_order[1]: "Medium Adoption",
        mapping_order[2]: "High Adoption"
    }
    
    cluster_data['Adoption Segment'] = cluster_data['Cluster_ID'].map(label_mapping)
    
    # Merge back to a copy of filtered data for visualization
    clustered_display = filtered_data.copy()
    clustered_display['Adoption Segment'] = "Unknown"
    clustered_display.loc[cluster_data.index, 'Adoption Segment'] = cluster_data['Adoption Segment']

    fig_cluster = px.scatter_3d(
        clustered_display, 
        x='total_vehicles_sold', 
        y='ev_market_share', 
        z='electric_vehicles_sold',
        color='Adoption Segment',
        color_discrete_map={
            "High Adoption": COLORS['primary'],
            "Medium Adoption": COLORS['secondary'],
            "Low Adoption": COLORS['quaternary'],
            "Unknown": COLORS['slate']
        },
        title="3D Scatter Plot of Adoption Segments",
        template=PLOTLY_THEME
    )
    
    # Adjust layout for better viewing
    fig_cluster.update_layout(scene=dict(
        xaxis_title='Total Vehicles Sold',
        yaxis_title='EV Market Share',
        zaxis_title='EVs Sold'
    ), margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.subheader("Cluster Distribution")
    cluster_counts = clustered_display['Adoption Segment'].value_counts().reset_index()
    fig_cluster_bar = px.bar(
        cluster_counts, x='Adoption Segment', y='count',
        color='Adoption Segment',
        color_discrete_map={
            "High Adoption": COLORS['primary'],
            "Medium Adoption": COLORS['secondary'],
            "Low Adoption": COLORS['quaternary']
        },
        template=PLOTLY_THEME,
        title="Number of Records per Adoption Segment"
    )
    st.plotly_chart(fig_cluster_bar, use_container_width=True)

else:
    st.warning("Not enough data points to perform clustering. Please select broader filters.")


# --- Insights and Interpretation ---
st.header("5. Insights and Interpretation", divider="blue")

with st.container():
    st.markdown(f"""
    ### Key Findings
    
    1. **High-Performing Regions:** Based on the current filters, states/regions with the highest number of EV sales are visualized in the *State-wise EV Sales Comparison*. These regions represent strong market penetration.
    2. **Emerging EV Markets:** Look at the *Adoption Segment* map. The **Medium Adoption** segment often points to emerging markets that are scaling rapidly. Areas starting to shift from Low to Medium adoption over the years (seen if adjusting the Year filter) indicate emerging momentum.
    3. **Trends Over Time:** The *Year-wise* and *Month-wise* plots showcase the trajectory of the market. Often, you will see exponential growth as consumer awareness and infrastructure increase.
    4. **Cluster Meaning in Simple Terms:**
       - <span style="color:{COLORS['primary']};font-weight:bold;">High Adoption:</span> Characterized by very high total EV sales and usually significant market share. These are mature or extremely successful EV markets.
       - <span style="color:{COLORS['secondary']};font-weight:bold;">Medium Adoption:</span> Average EV sales. These markets are growing. They have steady total vehicle sales and increasing EV share.
       - <span style="color:{COLORS['quaternary']};font-weight:bold;">Low Adoption:</span> Low EV sales volumes. These are either very small markets overall or markets where EVs have not yet penetrated significantly.
    """, unsafe_allow_html=True)


# --- Pattern Mining and Association Rules ---
st.header("6. Pattern Mining and Association Rules", divider="blue")

st.markdown("""
Discover underlying **if-then** relationships between regions, vehicle categories, time periods, and sales volumes using Association Rule Mining.
""")

@st.cache_data
def prepare_transaction_data(df):
    if df.empty:
        return pd.DataFrame()
        
    tdf = pd.DataFrame()
    tdf['State'] = "State_" + df['state'].astype(str)
    tdf['Category'] = "Cat_" + df['vehicle_category'].astype(str)
    tdf['Year'] = "Year_" + df['year'].astype(str)
    
    # Discretize EV sales using qcut if enough unique values exist
    try:
        tdf['Sales_Level'] = "Sales_" + pd.qcut(df['electric_vehicles_sold'].rank(method='first'), q=3, labels=['Low', 'Medium', 'High']).astype(str)
    except Exception as e:
        tdf['Sales_Level'] = "Sales_Unknown"
        
    try:
        tdf['Market_Share'] = "Share_" + pd.qcut(df['ev_market_share'].rank(method='first'), q=3, labels=['Low', 'Medium', 'High']).astype(str)
    except:
        tdf['Market_Share'] = "Share_Unknown"
    
    # One-hot encode using get_dummies to simulate transactions format
    basket = pd.get_dummies(tdf)
    # Ensure boolean
    basket = basket.astype(bool)
    return basket

with st.spinner("Preparing transaction data..."):
    basket = prepare_transaction_data(filtered_data)

col_mining1, col_mining2 = st.columns([1, 3])

with col_mining1:
    st.subheader("Mining Parameters")
    min_support = st.slider("Minimum Support", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
    min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    algorithm = st.radio("Algorithm", options=["FP-Growth", "Apriori"])
    run_mining = st.button("Generate Rules")

with col_mining2:
    if run_mining and not basket.empty:
        st.subheader("Discovered Association Rules")
        with st.spinner(f"Running {algorithm}..."):
            try:
                if algorithm == "Apriori":
                    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
                else:
                    frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
                
                if frequent_itemsets.empty:
                    st.warning("No frequent itemsets found. Try lowering the minimum support.")
                else:
                    # MLextend calculates rules
                    try:
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))
                    except TypeError:
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                    
                    if rules.empty:
                        st.warning("No association rules found. Try lowering the minimum confidence.")
                    else:
                        # Clean up rule display
                        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        # Filter to only show strong/meaningful columns
                        display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False)
                        st.dataframe(display_rules, use_container_width=True)
                        
                        st.subheader("Rule Strength Visualization")
                        fig_rules = px.scatter(
                            display_rules, x='support', y='confidence',
                            color='lift',
                            size='lift',
                            hover_data=['antecedents', 'consequents'],
                            title="Support vs. Confidence (Colored by Lift)",
                            template=PLOTLY_THEME,
                            color_continuous_scale='Sunset'
                        )
                        st.plotly_chart(fig_rules, use_container_width=True)
            except Exception as e:
                st.error(f"Error during mining: {e}")
    elif not run_mining:
        st.info("Adjust the parameters and click 'Generate Rules' to discover patterns.")
    elif basket.empty:
        st.error("Not enough data to mine patterns.")

# --- AI Chatbot Strategy Advisor ---
st.header("7. AI EV Strategy Advisor", divider="blue")

st.markdown("""
Consult the AI Advisor for real-time strategic decision making regarding Electric Vehicle adoption, infrastructure planning, and market insights.
""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

user_input = st.chat_input("Ask for EV strategy, market insights, or data interpretation...")

if user_input:
    # Render user msg
    with st.chat_message("user"):
        st.markdown(user_input, unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Call Groq API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Prepare context dynamically from data
        total_sales_context = f"{total_ev_sales:,.0f}"
        avg_share_context = f"{avg_market_share:.2f}%"
        
        system_prompt = f"You are an expert Strategic Consultant for Electric Vehicle (EV) adoption and data analysis. Currently, the dashboard shows {total_sales_context} total EVs sold and a {avg_share_context} average market share based on current filters. Provide professional, insightful, and strictly data-driven business strategy advice."
        
        payload = {
            "model": "llama-3.1-8b-instant", 
            "messages": [
                {"role": "system", "content": system_prompt}
            ] + st.session_state.chat_history[-5:], # send last 5 messages for brief context
            "temperature": 0.5,
            "max_tokens": 1024
        }
        
        # Pull key from streamlit secrets (this ensures it isn't published to Github)
        api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content']
                message_placeholder.markdown(answer, unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"API Error: {response.text}"
                message_placeholder.markdown(error_msg, unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        except Exception as e:
            message_placeholder.markdown(f"Request failed: {str(e)}", unsafe_allow_html=True)
