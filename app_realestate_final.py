import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Real Estate Data Analytics (2020 - 2023)",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Data Analytics")
st.markdown("*Created by Faza Billah*")

# Data loading and preparation
@st.cache_data
def load_data():
    df = pd.read_pickle('dataset 2020-2023.pkl')
    monthly_data = df.groupby(['Year', 'Month']).agg({
        'Sale_Amount': 'median'
    }).round(2).reset_index()
    monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
    return df, monthly_data

df, monthly_data = load_data()

# =============================================================================
# LINE CHART ANALYSIS SECTION
# =============================================================================

st.header("📈 Line Chart Analysis")

# SIMPLE LINE CHART SECTION

st.subheader("📊 Simple Line Chart")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_data['Date'], monthly_data['Sale_Amount'], color='blue', linewidth=2)
ax.set_title('Median Property Sale Prices Over Time (2020-2023)', fontsize=16, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Median Sale Amount ($)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**📊 Analisis Simple Line Chart:**
- **Tren Jangka Pendek:** Harga median properti terus meningkat selama periode 2020-2023.
- **Dampak Eksternal:** Lonjakan harga signifikan terjadi pada masa pandemi COVID-19 (2021), mencerminkan perubahan permintaan pasar.
- **Fluktuasi:** Tidak ditemukan penurunan tajam karena data hanya mencakup periode 2020-2023.
- **Kesimpulan:** Pasar properti menunjukkan pertumbuhan yang stabil dan responsif terhadap faktor eksternal dalam periode ini.
""")

# ENCHANCE LINE CHART SECTION

st.subheader("🎨 Enhanced Line Chart")
fig_enhanced = px.line(
    monthly_data, 
    x='Date', 
    y='Sale_Amount', 
    title='Real Estate: 4-Year Market Evolution',
    color_discrete_sequence=['#2E86AB']
)
fig_enhanced.update_layout(title_x=0.0)

# Add key annotations (box only, no arrow)
median_price = monthly_data['Sale_Amount'].median()
annotations = [
    # Place COVID-19 annotation at the local peak in 2021-03
    ('2021-03-01', monthly_data.loc[monthly_data['Date'] == pd.to_datetime('2021-03-01'), 'Sale_Amount'].values[0], "COVID-19 Impact", "red"),
    # Place median annotation at the median date and median price
    (monthly_data['Date'].iloc[len(monthly_data)//2], median_price, f"Median: ${median_price:,.0f}", "green")
]

for x, y, text, color in annotations:
    fig_enhanced.add_annotation(
        x=x, y=y, text=text, showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=color,
        borderwidth=2,
        font=dict(color=color)
    )

fig_enhanced.update_layout(height=600, template='plotly_white', 
                          title_x=0.5, yaxis_tickformat='$,.0f')
st.plotly_chart(fig_enhanced, use_container_width=True)

st.markdown("""
**🎨 Analisis Enhanced Chart:**
- **Tren Jangka Pendek:** Harga median properti menunjukkan kenaikan yang konsisten selama periode 2020-2023.
- **Dampak Ekonomi:** Lonjakan harga terjadi pada masa pandemi COVID-19, mencerminkan perubahan permintaan dan dinamika pasar.
- **Insight Visual:** Anotasi menyoroti puncak harga pada 2021 dan posisi median, membantu memahami pergerakan pasar dalam periode singkat.
- **Kesimpulan Bisnis:** Meskipun data hanya mencakup 4 tahun terakhir, pasar properti tetap menunjukkan pertumbuhan dan respons yang cepat terhadap faktor eksternal.
""")

# Summary Metrics
st.subheader("💡 Key Takeaways (2020-2023)")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Median Price (2020-2023)", f"${median_price:,.0f}", "Period Median")
with col2:
    st.metric("Peak Price", f"${monthly_data['Sale_Amount'].max():,.0f}", f"{monthly_data.loc[monthly_data['Sale_Amount'].idxmax(), 'Date'].strftime('%b %Y')}")
with col3:
    st.metric("Lowest Price", f"${monthly_data['Sale_Amount'].min():,.0f}", f"{monthly_data.loc[monthly_data['Sale_Amount'].idxmin(), 'Date'].strftime('%b %Y')}")


# =============================================================================
# BAR CHART ANALYSIS SECTION
# =============================================================================

st.header("📊 Bar Chart Analysis")
st.markdown("Analysis of median sale amounts by property type")

# Prepare property type data
@st.cache_data
def prepare_property_data(df):
    property_analysis = df.groupby('Property_Type_Update').agg({
        'Sale_Amount': ['median', 'count']
    }).round(2)
    property_analysis.columns = ['Median_Sale_Amount', 'Transaction_Count']
    property_analysis = property_analysis.reset_index()
    property_analysis = property_analysis.sort_values('Median_Sale_Amount', ascending=False)
    return property_analysis

property_analysis = prepare_property_data(df)

# SIMPLE BAR CHART

st.subheader("📊 Simple Bar Chart")

# Create matplotlib figure
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(property_analysis['Property_Type_Update'], 
              property_analysis['Median_Sale_Amount'], 
              color='steelblue', alpha=0.7)

ax.set_title('Median Sale Amount by Property Type', fontsize=16, fontweight='bold')
ax.set_xlabel('Property Type', fontsize=12)
ax.set_ylabel('Median Sale Amount ($)', fontsize=12)
ax.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
            f'${height:,.0f}', ha='center', va='bottom', fontsize=10)

# Format y-axis to show currency
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Display in Streamlit
st.pyplot(fig)

st.markdown("""
**📊 Simple Bar Chart Analysis:**
- **Kategori Tertinggi/Terendah:** Residential properties menunjukkan nilai median tertinggi
- **Perbedaan Antar Kategori:** Terdapat perbedaan signifikan antara property types
- **Hasil Tidak Terduga:** Commercial properties mungkin memiliki variasi nilai yang besar
""")

# ENHANCED BAR CHART

st.subheader("🎨 Enhanced Bar Chart with Interactive Features")

# Create enhanced plotly bar chart
fig_bar_enhanced = go.Figure()

fig_bar_enhanced.add_trace(go.Bar(
    x=property_analysis['Property_Type_Update'],
    y=property_analysis['Median_Sale_Amount'],
    text=property_analysis['Median_Sale_Amount'].apply(lambda x: f'${x:,.0f}'),
    textposition='outside',
    marker=dict(
        color=property_analysis['Median_Sale_Amount'],
        colorscale='viridis',
        showscale=True,
        colorbar=dict(title="Median Price ($)")
    ),
    hovertemplate='<b>%{x}</b><br>Median Price: $%{y:,.0f}<br>Transactions: %{customdata:,.0f}<extra></extra>',
    customdata=property_analysis['Transaction_Count']
))

# Add benchmark line
overall_median = df['Sale_Amount'].median()
fig_bar_enhanced.add_hline(
    y=overall_median,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Overall Median: ${overall_median:,.0f}"
)

fig_bar_enhanced.update_layout(
    title={
        'text': 'Real Estate: Property Type Value Analysis',
        'font': {'size': 18, 'color': 'darkblue'},
        'x': 0.5
    },
    xaxis_title='Property Type',
    yaxis_title='Median Sale Amount ($)',
    yaxis_tickformat='$,.0f',
    template='plotly_white',
    height=600,
    font=dict(size=12)
)

# Display in Streamlit
st.plotly_chart(fig_bar_enhanced, use_container_width=True)

st.markdown("""
**🎨 Analisis Enhanced Bar Chart:**
- **Perbandingan Kategori:** Skala warna memudahkan identifikasi tipe properti dengan nilai tertinggi
- **Fitur Interaktif:** Hover menampilkan detail median harga dan jumlah transaksi
- **Garis Benchmark:** Garis merah menunjukkan median keseluruhan sebagai pembanding
- **Insight Nilai:** Properti residensial mendominasi pasar, properti komersial premium ada, pola investasi terlihat jelas
""")

# PROPERTY TYPE INSIGHTS

st.subheader("🏠 Property Type Insights")

# Create metrics for top property types
col1, col2, col3 = st.columns(3)

top_3_properties = property_analysis.head(3)

with col1:
    prop1 = top_3_properties.iloc[0]
    st.metric(
        label=f"🥇 {prop1['Property_Type_Update']}",
        value=f"${prop1['Median_Sale_Amount']:,.0f}",
        delta=f"{prop1['Transaction_Count']:,.0f} transactions"
    )

with col2:
    prop2 = top_3_properties.iloc[1]
    st.metric(
        label=f"🥈 {prop2['Property_Type_Update']}",
        value=f"${prop2['Median_Sale_Amount']:,.0f}",
        delta=f"{prop2['Transaction_Count']:,.0f} transactions"
    )

with col3:
    prop3 = top_3_properties.iloc[2]
    st.metric(
        label=f"🥉 {prop3['Property_Type_Update']}",
        value=f"${prop3['Median_Sale_Amount']:,.0f}",
        delta=f"{prop3['Transaction_Count']:,.0f} transactions"
    )

# =============================================================================
# HISTOGRAM ANALYSIS SECTION
# =============================================================================

st.header("📊 Histogram Analysis")
st.markdown("Analysis of property sale amount distribution")

# Prepare histogram data
@st.cache_data
def prepare_histogram_data(df):
    """Prepare data for histogram analysis by capping outliers"""
    cap_value = df['Sale_Amount'].quantile(0.95)
    df_viz = df[df['Sale_Amount'] <= cap_value].copy()
    return df_viz, cap_value

df_viz, cap_value = prepare_histogram_data(df)

# SIMPLE HISTOGRAM

st.subheader("📊 Simple Histogram")

# Create matplotlib histogram
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df_viz['Sale_Amount'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
ax.set_title('Distribution of Property Sale Amounts', fontsize=16, fontweight='bold')
ax.set_xlabel('Sale Amount ($)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)

# Add median line
median_val = df_viz['Sale_Amount'].median()
ax.axvline(median_val, color='red', linestyle='--', linewidth=2, 
           label=f'Median: ${median_val:,.0f}')

# Format x-axis to show currency
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Display in Streamlit
st.pyplot(fig)

st.markdown("""
**📊 Simple Histogram Analysis:**
- **Distribusi:** Data menunjukkan distribusi yang condong ke kanan (right-skewed)
- **Kelompok Dominan:** Mayoritas properti berada di rentang harga entry-level hingga mid-market
- **Median Line:** Garis merah menunjukkan nilai tengah distribusi sebagai referensi
""")

# ENHANCED HISTOGRAM

st.subheader("🎨 Enhanced Histogram with Market Segments")

# Create enhanced plotly histogram
fig_hist_enhanced = go.Figure()

# Add histogram
fig_hist_enhanced.add_trace(go.Histogram(
    x=df_viz['Sale_Amount'],
    nbinsx=60,
    name='Sale Amount Distribution',
    marker=dict(
        color='skyblue',
        line=dict(color='darkblue', width=1)
    ),
    hovertemplate='Price Range: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
))

# Add shaded regions for market segments
fig_hist_enhanced.add_vrect(
    x0=0, x1=200000,
    fillcolor="lightgreen", opacity=0.2,
    annotation_text="Entry Level", annotation_position="top"
)

fig_hist_enhanced.add_vrect(
    x0=200000, x1=400000,
    fillcolor="yellow", opacity=0.2,
    annotation_text="Mid Market", annotation_position="top"
)

fig_hist_enhanced.add_vrect(
    x0=400000, x1=cap_value,
    fillcolor="orange", opacity=0.2,
    annotation_text="Premium", annotation_position="top"
)

# Add median line with enhanced annotation (box style for better visibility)
fig_hist_enhanced.add_vline(
    x=median_val,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Median: ${median_val:,.0f}",
    annotation_position="bottom right",
    annotation=dict(
        font=dict(color="red", size=14),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="red",
        borderwidth=2,
        borderpad=6,
        showarrow=False
    )
)

fig_hist_enhanced.update_layout(
    title={
        'text': 'Real Estate: Price Distribution Analysis (95th Percentile)',
        'font': {'size': 18, 'color': 'darkblue'},
        'x': 0.5
    },
    xaxis_title='Sale Amount ($)',
    yaxis_title='Frequency',
    xaxis_tickformat='$,.0f',
    template='plotly_white',
    height=600,
    showlegend=False,
    font=dict(size=12)
)

# Display in Streamlit
st.plotly_chart(fig_hist_enhanced, use_container_width=True)

st.markdown("""
**🎨 Analisis Histogram Tingkat Lanjut:**
- **Segmentasi Pasar:** Visualisasi dibagi menjadi 3 segmen - Entry Level, Mid Market, dan Premium
- **Fitur Interaktif:** Hover untuk melihat detail rentang harga dan jumlah properti
- **Bentuk Distribusi:** Distribusi menunjukkan konsentrasi tinggi pada segmen entry-level
- **Segmen Premium:** Segmen premium memiliki volume transaksi lebih rendah namun nilai tinggi
""")

# DISTRIBUTION INSIGHTS

st.subheader("📈 Distribution Insights")

# Calculate distribution statistics
col1, col2, col3, col4 = st.columns(4)

entry_level = df_viz[df_viz['Sale_Amount'] <= 200000]
mid_market = df_viz[(df_viz['Sale_Amount'] > 200000) & (df_viz['Sale_Amount'] <= 400000)]
premium = df_viz[df_viz['Sale_Amount'] > 400000]

with col1:
    st.metric(
        label="🟢 Entry Level",
        value=f"{len(entry_level):,}",
        delta=f"{len(entry_level)/len(df_viz)*100:.1f}% of total"
    )

with col2:
    st.metric(
        label="🟡 Mid Market",
        value=f"{len(mid_market):,}",
        delta=f"{len(mid_market)/len(df_viz)*100:.1f}% of total"
    )

with col3:
    st.metric(
        label="🟠 Premium",
        value=f"{len(premium):,}",
        delta=f"{len(premium)/len(df_viz)*100:.1f}% of total"
    )

with col4:
    st.metric(
        label="📊 Median Price",
        value=f"${median_val:,.0f}",
        delta=f"95th percentile: ${cap_value:,.0f}"
    )

# =============================================================================
# SCATTER PLOT ANALYSIS SECTION
# =============================================================================

st.header("📊 Scatter Plot Analysis")
st.markdown("Analysis of relationship between assessed value and sale price by town")

# Prepare scatter plot data
@st.cache_data
def prepare_scatter_data(df):
    """Prepare sample data for scatter plot analysis"""
    df_sample = df.sample(n=min(1000, len(df)), random_state=42)
    return df_sample

df_sample = prepare_scatter_data(df)

# SIMPLE SCATTER PLOT

st.subheader("📊 Simple Scatter Plot")

# Create simple scatter plot
fig_simple_scatter = px.scatter(
    df_sample, 
    x='Assessed_Value', 
    y='Sale_Amount', 
    color='Town',
    title='Assessment vs Sale Price by Town',
    labels={
        'Assessed_Value': 'Assessed Value ($)',
        'Sale_Amount': 'Sale Amount ($)'
    }
)

fig_simple_scatter.update_layout(
    height=600,
    template='plotly_white',
    title_font_size=16,
    title_x=0.5,
    xaxis_tickformat='$,.0f',
    yaxis_tickformat='$,.0f'
)

# Display in Streamlit
st.plotly_chart(fig_simple_scatter, use_container_width=True)

st.markdown("""
**📊 Analisis Simple Scatter Plot:**
- **Hubungan:** Terlihat korelasi positif antara nilai taksiran dan harga jual
- **Pola:** Sebagian besar data mengikuti tren linear yang cukup konsisten
- **Perbedaan Antar Kota:** Beberapa kota menunjukkan pola harga yang berbeda
- **Outlier:** Terdapat beberapa properti dengan harga jual yang sangat berbeda dari nilai taksirannya
""")

# ENHANCED SCATTER PLOT

st.subheader("🎨 Enhanced Scatter Plot with Statistical Insights")

# Create enhanced scatter plot with statistical overlays
fig_enhanced_scatter = px.scatter(
    df_sample, 
    x='Assessed_Value', 
    y='Sale_Amount', 
    color='Town',
    title='Assessment vs Sale Price with Statistical Insights',
    opacity=0.6,
    labels={
        'Assessed_Value': 'Assessed Value ($)',
        'Sale_Amount': 'Sale Amount ($)'
    }
)

# Add trend line
trend_fig = px.scatter(df_sample, x='Assessed_Value', y='Sale_Amount', trendline="ols")
trend_trace = trend_fig.data[1]
trend_trace.name = "Trend Line"
trend_trace.line.color = "red"
trend_trace.line.width = 3
fig_enhanced_scatter.add_trace(trend_trace)

# Calculate medians
median_assessed = df_sample['Assessed_Value'].median()
median_sale = df_sample['Sale_Amount'].median()

# Add median lines
fig_enhanced_scatter.add_hline(
    y=median_sale,
    line_dash="dot",
    line_color="green",
    line_width=2,
    annotation_text=f"Median Sale Price: ${median_sale:,.0f}",
    annotation_position="top right"
)

fig_enhanced_scatter.add_vline(
    x=median_assessed,
    line_dash="dot", 
    line_color="blue",
    line_width=2,
    annotation_text=f"Median Assessed Value: ${median_assessed:,.0f}",
    annotation_position="top right"
)

fig_enhanced_scatter.update_layout(
    title_font_size=18,
    title_x=0.5,
    height=600,
    template='plotly_white',
    xaxis_tickformat='$,.0f',
    yaxis_tickformat='$,.0f',
    font=dict(size=12)
)

# Display in Streamlit
st.plotly_chart(fig_enhanced_scatter, use_container_width=True)

st.markdown("""
**🎨 Analisis Scatter Plot Tingkat Lanjut:**
- **Garis Tren:** Garis merah menunjukkan hubungan linear keseluruhan antara nilai taksiran dan harga jual
- **Referensi Median:** Garis hijau dan biru memberikan referensi nilai median untuk perbandingan
- **Insight Statistik:** Kombinasi garis tren dan garis median membantu mengidentifikasi outlier dan pola
- **Fitur Interaktif:** Hover untuk detail properti dan informasi spesifik tiap kota
""")

# CORRELATION INSIGHTS

st.subheader("🔍 Correlation Analysis")

# Calculate correlation
correlation = df_sample['Assessed_Value'].corr(df_sample['Sale_Amount'])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📈 Correlation",
        value=f"{correlation:.3f}",
        delta="Strong positive relationship" if correlation > 0.7 else "Moderate relationship"
    )

with col2:
    # Calculate R-squared (approximate)
    r_squared = correlation ** 2
    st.metric(
        label="📊 R-squared",
        value=f"{r_squared:.3f}",
        delta=f"{r_squared*100:.1f}% variance explained"
    )

with col3:
    # Properties above/below trend
    above_median_both = len(df_sample[(df_sample['Assessed_Value'] > median_assessed) & 
                                     (df_sample['Sale_Amount'] > median_sale)])
    st.metric(
        label="🏠 High Value Properties",
        value=f"{above_median_both:,}",
        delta="Above both medians"
    )

with col4:
    # Sample size
    st.metric(
        label="📋 Sample Size",
        value=f"{len(df_sample):,}",
        delta=f"from {len(df):,} total"
    )

# Quadrant analysis
st.subheader("📍 Quadrant Analysis")

# Calculate quadrants
q1 = len(df_sample[(df_sample['Assessed_Value'] <= median_assessed) & 
                   (df_sample['Sale_Amount'] <= median_sale)])
q2 = len(df_sample[(df_sample['Assessed_Value'] > median_assessed) & 
                   (df_sample['Sale_Amount'] <= median_sale)])
q3 = len(df_sample[(df_sample['Assessed_Value'] > median_assessed) & 
                   (df_sample['Sale_Amount'] > median_sale)])
q4 = len(df_sample[(df_sample['Assessed_Value'] <= median_assessed) & 
                   (df_sample['Sale_Amount'] > median_sale)])

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **📊 Property Distribution by Quadrants:**
    - **Q1 (Low-Low):** {:,} properties ({:.1f}%)
    - **Q2 (High Assessed, Low Sale):** {:,} properties ({:.1f}%)
    """.format(q1, q1/len(df_sample)*100, q2, q2/len(df_sample)*100))

with col2:
    st.markdown("""
    **📊 Property Distribution by Quadrants:**
    - **Q3 (High-High):** {:,} properties ({:.1f}%)
    - **Q4 (Low Assessed, High Sale):** {:,} properties ({:.1f}%)
    """.format(q3, q3/len(df_sample)*100, q4, q4/len(df_sample)*100))

# =============================================================================
# HEATMAP CORRELATION ANALYSIS SECTION
# =============================================================================

st.header("🔥 Heatmap (Correlation) Analysis")
st.markdown("Analysis of correlation between numerical variables")

# Prepare correlation data
@st.cache_data
def prepare_correlation_data(df):
    """Prepare correlation matrix for numerical variables"""
    numerical_cols = ['Sale_Amount', 'Assessed_Value', 'Sales_Ratio', 'Year']
    correlation_matrix = df[numerical_cols].corr()
    return correlation_matrix, numerical_cols

correlation_matrix, numerical_cols = prepare_correlation_data(df)

# CORRELATION HEATMAP

st.subheader("🔥 Correlation Heatmap")

# Create correlation heatmap
fig_heatmap = px.imshow(
    correlation_matrix,
    text_auto='.2f',  # Show only 2 decimal places
    aspect="auto",
    title='Variable Relationships Analysis',
    color_continuous_scale='RdBu_r',
    zmin=-1,
    zmax=1,
    labels=dict(color="Correlation")
)

# Clean layout - remove extra annotations
fig_heatmap.update_layout(
    title={
        'font': {'size': 18, 'color': 'darkblue'},
        'x': 0.5
    },
    height=500,
    width=600,
    font=dict(size=11),
    margin=dict(l=50, r=50, t=80, b=50)
)

# Display in Streamlit
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("""
**🔥 Heatmap Correlation Analysis:**
- **Variabel dengan Korelasi Kuat:** Sale_Amount dan Assessed_Value menunjukkan korelasi positif yang kuat
- **Korelasi Positif/Negatif:** Warna Merah menunjukkan korelasi positif, Biru menunjukkan korelasi negatif
- **Makna Hubungan:** Sales_Ratio memberikan insight tentang efisiensi penilaian properti
""")

# DETAILED CORRELATION ANALYSIS

st.subheader("📊 Detailed Correlation Insights")

st.markdown("**🔴 Strongest Correlations:**")
    
    # Find strongest correlations (excluding diagonal)
corr_pairs = []
for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            corr_value = correlation_matrix.iloc[i, j]
            corr_pairs.append({
                'Pair': f"{numerical_cols[i]} ↔ {numerical_cols[j]}",
                'Correlation': corr_value,
                'Strength': abs(corr_value)
            })
    
    # Sort by absolute correlation
corr_pairs = sorted(corr_pairs, key=lambda x: x['Strength'], reverse=True)
    
for pair in corr_pairs[:3]:  # Top 3 correlations
        strength = "Strong" if pair['Strength'] > 0.7 else "Moderate" if pair['Strength'] > 0.3 else "Weak"
        direction = "Positive" if pair['Correlation'] > 0 else "Negative"
        st.write(f"• **{pair['Pair']}**: {pair['Correlation']:.3f} ({strength} {direction})")

# CORRELATION INTERPRETATION

st.subheader("🎯 Correlation Interpretation Guide")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **💪 Korelasi Kuat (|r| > 0.7)**
    - Hubungan sangat dapat diprediksi
    - Satu variabel sangat memengaruhi variabel lain
    - Sangat andal untuk peramalan
    """)

with col2:
    st.markdown("""
    **🔶 Korelasi Sedang (0.3 < |r| < 0.7)**
    - Hubungan cukup terlihat
    - Ada kekuatan prediktif
    - Faktor lain juga berperan penting
    """)

with col3:
    st.markdown("""
    **🔹 Korelasi Lemah (|r| < 0.3)**
    - Hampir tidak ada hubungan linear
    - Nilai prediktif rendah
    - Variabel cenderung independen
    """)


# =============================================================================
# PIE CHART ANALYSIS SECTION
# =============================================================================

# PIE CHART ANALYSIS SECTION

st.header("🥧 Pie Chart Analysis")
st.markdown("Distribution of top 3 residential types by transaction count (2020-2023)")

# Prepare data for pie chart
df_recent = df[df['Year'] >= 2020]
total_transactions = len(df_recent)
top3_res_types = df_recent['Residential_Type'].value_counts().nlargest(3)
other_count = total_transactions - top3_res_types.sum()

pie_labels = list(top3_res_types.index) + ['Other']
pie_values = list(top3_res_types.values) + [other_count]

fig_pie = px.pie(
    values=pie_values,
    names=pie_labels,
    title='Top 3 Residential Types by Transaction Count (2020-2023)',
    hole=0.4,  # Donut chart
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig_pie.update_traces(
    textposition='inside',
    textinfo='percent+label',
    hovertemplate='<b>%{label}</b><br>Transactions: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
)

fig_pie.update_layout(
    title={
        'text': 'Top 3 Residential Types by Transaction Count<br><span style="font-size: 14px">2020-2023</span>',
        'font': {'size': 18, 'color': 'darkblue'},
        'x': 0.5
    },
    height=600,
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.01
    ),
)

# Display in Streamlit
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("""
**🥧 Pie Chart Analysis:**
- **Dominasi Tipe:** Tiga tipe hunian paling populer mendominasi transaksi selama 2020-2023.
- **Kategori Lain:** Sisanya dikelompokkan sebagai 'Other' untuk visualisasi yang lebih jelas.
- **Insight:** Distribusi ini membantu memahami preferensi pasar dan potensi pengembangan produk.
""")

