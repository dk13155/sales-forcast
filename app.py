import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from prophet import Prophet
from datetime import datetime



# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="📊 Interactive Sales Forecast Dashboard",
                   page_icon="📈", layout="wide")

# ---------------------------
# CSS + Background + Cards
# ---------------------------
st.markdown("""
<style>
/* Hide Streamlit default menus */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Gradient animated background */
body {
    background: linear-gradient(120deg, #f093fb, #f5576c, #4facfe, #00f2fe);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Header card */
.header-card {
    background: linear-gradient(90deg,#ff6a00,#ee0979);
    padding: 25px; border-radius: 20px;
    text-align:center; color:white;
    box-shadow:0 10px 25px rgba(0,0,0,0.4);
}

/* KPI cards */
.kpi-card {
    padding: 20px; border-radius:15px; color:white;
    text-align:center; position:relative; overflow:hidden;
    font-weight:bold; box-shadow:0 8px 20px rgba(0,0,0,0.3);
}
.kpi-title {font-size:18px;margin-bottom:10px;}
.kpi-value {font-size:28px;color:#fff;}
.kpi-bar {
    position:absolute; bottom:0; left:0;
    width:100%; height:0%; border-radius:15px 15px 0 0;
    animation: growBar 1.5s forwards;
}
@keyframes growBar { from {height:0%} to {height: var(--bar-height);} }

/* Info cards */
.text-card {
    background: linear-gradient(120deg,#f6d365,#fda085);
    padding: 20px; border-radius:15px; color:white;
    margin-bottom:20px; box-shadow:0 8px 20px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header Section
# ---------------------------
st.markdown(
    '<div class="header-card"><h1>📈 Interactive Sales Forecast Dashboard</h1><p> dashboard with animated KPIs, top products, cumulative sales, trends, and live forecasting.</p></div>',
    unsafe_allow_html=True
)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["Historical Analysis", "Live Forecast"])

# ---------------------------
# Tab 1: Historical Analysis
# ---------------------------
with tab1:
   
        data = pd.read_csv("retail_sales_dataset.csv", parse_dates=['Date'])
        
        # --- Date range filter ---
        min_date, max_date = st.date_input("Select Date Range", [data['Date'].min(), data['Date'].max()])
        data = data[(data['Date']>=pd.to_datetime(min_date)) & (data['Date']<=pd.to_datetime(max_date))]
        
        # --- Aggregation ---
        agg_option = st.radio("Aggregate sales by:", ["Daily","Weekly","Monthly"], horizontal=True)
        if agg_option=="Daily":
            sales_data = data.groupby('Date')['Total Amount'].sum().reset_index()
        elif agg_option=="Weekly":
            sales_data = data.resample('W-Mon', on='Date')['Total Amount'].sum().reset_index()
        else:
            sales_data = data.resample('M', on='Date')['Total Amount'].sum().reset_index()
        sales_data.rename(columns={'Date':'ds','Total Amount':'y'}, inplace=True)
        
        # --- Calculate % Change ---
        sales_data['pct_change'] = sales_data['y'].pct_change()*100

        # --- Animated KPI Cards ---
        total_sales = sales_data['y'].sum()
        avg_sales = sales_data['y'].mean()
        max_sales = sales_data['y'].max()
        min_sales = sales_data['y'].min()
        col1,col2,col3,col4 = st.columns(4)
        kpi_info = [("Total Sales",total_sales,"linear-gradient(90deg,#f7971e,#ffd200)"),
                    ("Average Sales",avg_sales,"linear-gradient(90deg,#00c6ff,#0072ff)"),
                    ("Max Sales",max_sales,"linear-gradient(90deg,#ff5f6d,#ffc371)"),
                    ("Min Sales",min_sales,"linear-gradient(90deg,#36d1dc,#5b86e5)")]
        for col,(title,value,color) in zip([col1,col2,col3,col4],kpi_info):
            col.markdown(f"""
            <div class="kpi-card" style="background:{color};">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">${value:,.0f}</div>
                <div class="kpi-bar" style="--bar-height:{(value/total_sales*100 if total_sales>0 else 50)}%; background: rgba(255,255,255,0.3);"></div>
            </div>
            """, unsafe_allow_html=True)

        # --- Sales Trend Chart ---
        st.subheader(f"📈 {agg_option} Sales Trend with % Change")
        fig = px.line(sales_data, x='ds', y='y', template='plotly_white', markers=True)
        fig.add_trace(go.Scatter(x=sales_data['ds'], y=sales_data['y'], mode='lines+markers', name='Sales', line=dict(color='#ff6a00', width=4)))
        # Add % change arrows
        for i,row in sales_data.iterrows():
            if i==0: continue
            arrow = "🔺" if row['pct_change']>0 else "🔻"
            fig.add_annotation(x=row['ds'], y=row['y'], text=f"{arrow}{abs(row['pct_change']):.1f}%", showarrow=True, arrowhead=1)
        st.plotly_chart(fig,use_container_width=True)

        # --- Cumulative Sales Chart ---
        st.subheader("📊 Cumulative Sales")
        sales_data['cum_sales'] = sales_data['y'].cumsum()
        fig_cum = px.area(sales_data, x='ds', y='cum_sales', color_discrete_sequence=['#36d1dc'])
        fig_cum.update_layout(template='plotly_white', yaxis_title='Cumulative Sales')
        st.plotly_chart(fig_cum,use_container_width=True)

        # --- Top N Products ---
        st.subheader("🏆 Top N Products")
        top_n = st.slider("Select top N products",3,10,5)
        top_products = data.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False).reset_index().head(top_n)
        fig2 = px.bar(top_products,x='Product Category',y='Total Amount',color='Total Amount',color_continuous_scale='Turbo')
        st.plotly_chart(fig2,use_container_width=True)
        
        # --- Pie Chart for Product Contribution ---
        st.subheader("🥧 Top Products Contribution")
        fig_pie = px.pie(top_products, names='Product Category', values='Total Amount', color_discrete_sequence=px.colors.sequential.Aggrnyl)
        st.plotly_chart(fig_pie,use_container_width=True)

        # --- Forecast Next 30 Days ---
        st.subheader("🔮 Forecast Next 30 Days")
        model = Prophet(daily_seasonality=True)
        model.fit(sales_data)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig3 = go.Figure()
        # progressive line animation (approximate using frames)
        fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#ff6a00', width=4)))
        fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='lightgrey', name='Upper CI'))
        fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='lightgrey', name='Lower CI'))
        fig3.update_layout(title='Forecast with Confidence Interval', yaxis_title='Sales', template='plotly_white')
        st.plotly_chart(fig3,use_container_width=True)

        forecast_csv = forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(index=False)
        st.download_button("Download Historical Forecast CSV", forecast_csv, file_name="historical_forecast.csv", mime="text/csv", key="hist_download")

# ---------------------------
# Tab 2: Live Forecast (Enhanced, Only This Tab)
# ---------------------------
with tab2:
    st.markdown(
        '<div class="text-card"><h2>⚡ Live Forecast Demonstration</h2>'
        '<p>Adjust forecast period, visualize dynamic predictions, simulate scenarios, and explore insights interactively.</p></div>', 
        unsafe_allow_html=True
    )

    # Load pre-trained model
    try:
        with open('prophet_sales_model.pkl','rb') as f:
            pre_trained_model = pickle.load(f)
    except:
        st.error("Pre-trained model not found! Please run `train_model.py` first.")

    if 'pre_trained_model' in locals():
        # --- Forecast period slider ---
        forecast_days = st.slider("Forecast Period (days)", 7, 180, 30, 7)
        
        # --- Scenario simulation sliders ---
        st.markdown("<h4>🔧 Scenario Simulation</h4>", unsafe_allow_html=True)
        promo_boost = st.slider("Promotion / Boost Factor (%)", -50, 100, 0, 5)
        season_adj = st.slider("Seasonality Adjustment (%)", -30, 30, 0, 5)
        
        # --- Options for showing history / CI ---
        show_history = st.checkbox("Show Historical Data", value=True)
        show_ci = st.checkbox("Show Confidence Interval", value=True)
        
        # --- Forecast calculation ---
        future_live = pre_trained_model.make_future_dataframe(periods=forecast_days)
        forecast_live = pre_trained_model.predict(future_live)
        adjustment_factor = 1 + (promo_boost/100) + (season_adj/100)
        forecast_live['yhat'] *= adjustment_factor
        forecast_live['yhat_upper'] *= adjustment_factor
        forecast_live['yhat_lower'] *= adjustment_factor
        forecast_live['pct_change'] = forecast_live['yhat'].pct_change()*100
        
        # --- Gradient KPI cards ---
        total_forecast = forecast_live['yhat'].sum()
        avg_forecast = forecast_live['yhat'].mean()
        max_forecast = forecast_live['yhat'].max()
        min_forecast = forecast_live['yhat'].min()
        col1, col2, col3, col4 = st.columns(4)
        kpi_info = [
            ("Total Forecast", total_forecast, "linear-gradient(90deg,#f7971e,#ffd200)"),
            ("Average Daily Forecast", avg_forecast, "linear-gradient(90deg,#00c6ff,#0072ff)"),
            ("Max Forecast", max_forecast, "linear-gradient(90deg,#ff5f6d,#ffc371)"),
            ("Min Forecast", min_forecast, "linear-gradient(90deg,#36d1dc,#5b86e5)")
        ]
        for col, (title, value, color) in zip([col1,col2,col3,col4], kpi_info):
            col.markdown(f"""
            <div class="kpi-card" style="background:{color};">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">${value:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # --- Interactive chart ---
        fig_live = go.Figure()
        if show_history:
            hist_data = pre_trained_model.history
            fig_live.add_trace(go.Scatter(
                x=hist_data['ds'], y=hist_data['y'], mode='lines+markers',
                name='Historical', line=dict(color='#ff6a00', width=3)
            ))
        fig_live.add_trace(go.Scatter(
            x=forecast_live['ds'], y=forecast_live['yhat'], mode='lines+markers',
            name='Forecast', line=dict(color='#00c6ff', width=4)
        ))
        if show_ci:
            fig_live.add_trace(go.Scatter(
                x=forecast_live['ds'], y=forecast_live['yhat_upper'], fill=None,
                mode='lines', line_color='lightgrey', name='Upper CI'
            ))
            fig_live.add_trace(go.Scatter(
                x=forecast_live['ds'], y=forecast_live['yhat_lower'], fill='tonexty',
                mode='lines', line_color='lightgrey', name='Lower CI'
            ))
        fig_live.update_layout(
            title=f"Live Forecast Next {forecast_days} Days",
            template='plotly_white',
            yaxis_title='Sales',
            hovermode='x unified'
        )
        st.plotly_chart(fig_live, use_container_width=True)
        
        # --- Forecast summary table (next 7 days) ---
        st.subheader("📄 Forecast Summary (Next 7 Days)")
        summary_table = forecast_live[['ds','yhat','yhat_lower','yhat_upper','pct_change']].head(7)
        summary_table = summary_table.rename(columns={'ds':'Date','yhat':'Forecast','yhat_lower':'Lower CI','yhat_upper':'Upper CI','pct_change':'% Change'})
        st.dataframe(summary_table.style.format({'Forecast':'${:,.0f}','Lower CI':'${:,.0f}','Upper CI':'${:,.0f}','% Change':'{:+.1f}%'}))
        
        # --- Top 3 Insights ---
        st.subheader("💡 Top Insights")
        max_day = forecast_live.loc[forecast_live['yhat'].idxmax(),'ds'].strftime('%Y-%m-%d')
        min_day = forecast_live.loc[forecast_live['yhat'].idxmin(),'ds'].strftime('%Y-%m-%d')
        total_cum = forecast_live['yhat'].sum()
        insights = [
            f"Highest predicted sales on: {max_day}",
            f"Lowest predicted sales on: {min_day}",
            f"Cumulative forecast sales: ${total_cum:,.0f}"
        ]
        for insight in insights:
            st.markdown(f'<div class="text-card">{insight}</div>', unsafe_allow_html=True)
        
        # --- Download live forecast CSV ---
        forecast_csv = forecast_live[['ds','yhat','yhat_lower','yhat_upper','pct_change']].to_csv(index=False)
        st.download_button(
            "Download Live Forecast CSV",
            forecast_csv,
            file_name="live_forecast.csv",
            mime="text/csv",
            key="live_download_final"
        )
        

