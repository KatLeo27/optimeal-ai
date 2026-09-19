"""
OptiMeal AI — AI-Powered Meal Demand Optimization & Food Waste Intelligence
Client-Ready, Bright & Lively UI for Institutional Food Service Operations.
1M1B AI for Sustainability Virtual Internship in collaboration with IBM SkillsBuild and AICTE.
"""

import sys
import os

# Ensure root path is configured
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_processed_data
from src.waste_analysis import (
    compute_executive_kpis,
    get_waste_by_meal_type,
    get_waste_by_food_category,
    get_waste_by_section,
    get_daily_waste_timeseries,
    get_highest_waste_days
)
from src.anomaly_detection import detect_statistical_anomalies
from src.demand_prediction import DemandPredictor
from src.recommendations import optimize_meal_preparation
from src.insights import generate_rule_based_insights, generate_ai_prediction_narrative
from src.rag import SustainabilityRAGAssistant
from src.utils import (
    CUSTOM_CSS,
    COLOR_EMERALD,
    COLOR_LEAF,
    COLOR_LIME,
    COLOR_AMBER,
    COLOR_YELLOW,
    COLOR_ORANGE,
    COLOR_CORAL,
    COLOR_SEQUENCE,
    format_kg,
    format_currency,
    format_pct,
    get_provenance_badge,
    apply_custom_plotly_layout
)

# Page configuration
st.set_page_config(
    page_title="OptiMeal AI — Meal Demand & Waste Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply bright, lively styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Load cached pipeline assets
@st.cache_data
def get_dataset():
    return load_processed_data()

@st.cache_resource
def get_predictor():
    return DemandPredictor()

@st.cache_resource
def get_rag():
    return SustainabilityRAGAssistant()

df = get_dataset()
predictor = get_predictor()
rag_assistant = get_rag()
kpis = compute_executive_kpis(df)

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 14px 0;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.6rem;">🌱</span>
                <div>
                    <h2 style="color: #065F46; margin: 0; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.01em;">OptiMeal AI</h2>
                    <p style="color: #D97706; font-size: 0.78rem; margin: 0; font-weight: 700; text-transform: uppercase;">Waste Intelligence Platform</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid #FED7AA;'>", unsafe_allow_html=True)
    
    selected_page = st.radio(
        "Navigation",
        [
            "🌟 Operations Overview",
            "🔮 Demand Prediction & Optimizer",
            "📊 Waste Intelligence & Outliers",
            "🧪 What-If Scenario Simulator",
            "🧠 Automated AI Insights",
            "📚 Sustainability Assistant (RAG)"
        ],
        index=0
    )

# ---------------------------------------------------------
# PAGE 1: OPERATIONS OVERVIEW
# ---------------------------------------------------------
if selected_page == "🌟 Operations Overview":
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">Food Service Intelligence & Demand Optimization</h1>
            <p class="hero-subtitle">Real-time operational dashboard for corporate cafeterias, universities, hospitals & hotels.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Executive KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card-green">
                <div class="metric-label">Total Waste Weight</div>
                <div class="metric-value" style="color: #047857;">{format_kg(kpis['total_waste_kg'])}</div>
                <div class="metric-subtext" style="color: #059669;">Across 61 service days</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card-orange">
                <div class="metric-label">Financial Cost Loss</div>
                <div class="metric-value" style="color: #C2410C;">{format_currency(kpis['total_cost_loss'])}</div>
                <div class="metric-subtext" style="color: #EA580C;">{format_currency(kpis['avg_daily_cost_loss'])} / day avg</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card-yellow">
                <div class="metric-label">Surplus Overproduction</div>
                <div class="metric-value" style="color: #B45309;">{kpis['surplus_rate_pct']}%</div>
                <div class="metric-subtext" style="color: #D97706;">{kpis['total_surplus_meals']:,} unserved portions</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card-lime">
                <div class="metric-label">Total Meals Served</div>
                <div class="metric-value" style="color: #4D7C0F;">{kpis['total_meals_consumed']:,}</div>
                <div class="metric-subtext" style="color: #65A30D;">of {kpis['total_meals_prepared']:,} prepared</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # High-impact Charts Row
    c_left, c_right = st.columns([1.6, 1.0])
    with c_left:
        daily_ts = get_daily_waste_timeseries(df)
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Bar(
            x=daily_ts["Date"],
            y=daily_ts["daily_waste_kg"],
            name="Daily Food Waste (kg)",
            marker_color="#FDBA74",
            opacity=0.7
        ))
        fig_ts.add_trace(go.Scatter(
            x=daily_ts["Date"],
            y=daily_ts["waste_kg_7d_mavg"],
            mode="lines",
            name="7-Day Moving Baseline",
            line={"color": COLOR_EMERALD, "width": 3.5}
        ))
        fig_ts = apply_custom_plotly_layout(fig_ts, "Daily Waste Weight (kg) & 7-Day Trend", height=340)
        st.plotly_chart(fig_ts, use_container_width=True)

    with c_right:
        cat_df = get_waste_by_food_category(df)
        fig_cat = px.pie(
            cat_df,
            names="Food_Category",
            values="total_cost_loss",
            title="Financial Loss ($) by Food Stream",
            color="Food_Category",
            color_discrete_sequence=[COLOR_ORANGE, COLOR_EMERALD, COLOR_AMBER, COLOR_LIME],
            hole=0.45
        )
        fig_cat = apply_custom_plotly_layout(fig_cat, "Financial Loss ($) Share", height=340)
        st.plotly_chart(fig_cat, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: DEMAND PREDICTION & OPTIMIZER
# ---------------------------------------------------------
elif selected_page == "🔮 Demand Prediction & Optimizer":
    st.markdown("## 🔮 Intelligent Demand Prediction & Kitchen Optimizer")
    st.markdown("Forecast exact consumed portions and generate a data-driven preparation recommendation with dynamic safety buffers.")
    
    col_input, col_result = st.columns([1.1, 1.3])
    
    with col_input:
        st.markdown("### 📋 Planned Shift Inputs")
        with st.container():
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                meal_in = st.selectbox("Meal Service Shift", ["Lunch", "Dinner", "Breakfast"], index=0)
                cat_in = st.selectbox("Food Category", ["Vegetables", "Rice", "Meat", "Soup"], index=0)
                sec_in = st.selectbox("Canteen Section", ["A", "B", "C", "D"], index=0)
                day_in = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=2)
            with f_col2:
                exp_att = st.number_input("Expected Attendance", min_value=50, max_value=1500, value=620, step=10)
                prev_cons = st.number_input("Prev Day Consumption", min_value=40, max_value=1500, value=580, step=10)
                temp_in = st.slider("Temperature (°C)", 15.0, 42.0, 29.0, 0.5)
                rain_in = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 1.0)
            
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                event_in = st.checkbox("Special Event", value=False)
            with c_e2:
                holiday_in = st.checkbox("Public Holiday", value=False)
                
            calc_btn = st.button("⚡ Calculate Optimal Kitchen Target", use_container_width=True)

    # Perform inference
    is_wknd = 1 if day_in in ["Saturday", "Sunday"] else 0
    payload = {
        "Meal": meal_in,
        "Food_Category": cat_in,
        "Canteen_Section": sec_in,
        "Day_of_Week": day_in,
        "expected_attendance": exp_att,
        "temperature": temp_in,
        "rainfall": rain_in,
        "holiday": int(holiday_in),
        "special_event": int(event_in),
        "previous_day_consumption": prev_cons,
        "Is_Weekend": is_wknd,
        "Month": 7,
        "Day": 15
    }
    
    pred_res = predictor.predict_demand(payload)
    rec_res = optimize_meal_preparation(
        predicted_demand=pred_res["predicted_consumption"],
        expected_attendance=exp_att,
        meal_type=meal_in,
        food_category=cat_in,
        special_event=int(event_in),
        holiday=int(holiday_in),
        rainfall=rain_in
    )
    
    with col_result:
        st.markdown("### 🎯 Recommended Target & Savings")
        
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(
                f"""
                <div class="metric-card-green">
                    <div class="metric-label">Predicted Consumption</div>
                    <div class="metric-value" style="color: #047857;">{pred_res['predicted_consumption']:,} portions</div>
                    <div class="metric-subtext" style="color: #059669;">90% Range: {pred_res['lower_bound_90pct']} – {pred_res['upper_bound_90pct']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r2:
            st.markdown(
                f"""
                <div class="metric-card-orange">
                    <div class="metric-label">Recommended Prep Volume</div>
                    <div class="metric-value" style="color: #C2410C;">{rec_res['recommended_preparation']:,} portions</div>
                    <div class="metric-subtext" style="color: #EA580C;">Dynamic Buffer: +{rec_res['optimized_buffer_pct']}% (vs 12% static)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        r3, r4 = st.columns(2)
        with r3:
            st.markdown(
                f"""
                <div class="metric-card-yellow">
                    <div class="metric-label">Estimated Waste Avoided</div>
                    <div class="metric-value" style="color: #B45309;">{rec_res['estimated_waste_kg_saved']} kg</div>
                    <div class="metric-subtext" style="color: #D97706;">~{rec_res['meals_saved_estimate']} surplus meals saved</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r4:
            st.markdown(
                f"""
                <div class="metric-card-lime">
                    <div class="metric-label">Stockout Risk Level</div>
                    <div class="metric-value" style="color: {rec_res['waste_risk_color']};">{rec_res['waste_risk_level']}</div>
                    <div class="metric-subtext" style="color: #4D7C0F;">Expected Surplus: {rec_res['expected_surplus']} meals</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown(
            f"""
            <div class="insight-card-green">
                <div class="card-title">💡 Operational Rationale</div>
                <div class="card-body">{rec_res['operational_rationale']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# PAGE 3: WASTE INTELLIGENCE & OUTLIERS
# ---------------------------------------------------------
elif selected_page == "📊 Waste Intelligence & Outliers":
    st.markdown("## 📊 Granular Waste Analytics & Anomaly Detection")
    st.markdown("Pinpoint cost drivers and statistical anomaly days across canteen stations.")
    
    t1, t2 = st.columns(2)
    with t1:
        cat_df = get_waste_by_food_category(df)
        fig_c = px.bar(
            cat_df,
            x="Food_Category",
            y=["total_waste_kg", "total_cost_loss"],
            barmode="group",
            title="Waste Weight (kg) vs Financial Cost Loss ($)",
            color_discrete_sequence=[COLOR_EMERALD, COLOR_ORANGE]
        )
        fig_c = apply_custom_plotly_layout(fig_c, "Weight (kg) vs Financial Loss ($)", height=320)
        st.plotly_chart(fig_c, use_container_width=True)
        
    with t2:
        meal_df = get_waste_by_meal_type(df)
        fig_m = px.bar(
            meal_df,
            x="Meal",
            y="total_waste_kg",
            color="Meal",
            title="Total Waste Weight by Meal Shift",
            color_discrete_sequence=[COLOR_AMBER, COLOR_EMERALD, COLOR_ORANGE]
        )
        fig_m = apply_custom_plotly_layout(fig_m, "Total Waste (kg) by Shift", height=320)
        st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("### ⚠️ Top Operational Waste Outlier Days")
    high_days = get_highest_waste_days(df, 8)
    st.dataframe(
        high_days[[
            "Date", "Day_of_Week", "total_waste_kg", "total_cost_loss",
            "meals_prepared", "meals_consumed", "surplus_rate_pct"
        ]].rename(columns={
            "Date": "Date",
            "Day_of_Week": "Day",
            "total_waste_kg": "Waste (kg)",
            "total_cost_loss": "Cost ($)",
            "meals_prepared": "Prepared",
            "meals_consumed": "Consumed",
            "surplus_rate_pct": "Surplus (%)"
        }),
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------
# PAGE 4: WHAT-IF SCENARIO SIMULATOR
# ---------------------------------------------------------
elif selected_page == "🧪 What-If Scenario Simulator":
    st.markdown("## 🧪 What-If Scenario Simulator")
    st.markdown("Simulate how weather shifts, attendance surges, and custom kitchen prep policies impact financial costs and carbon emissions.")
    
    col_sim_in, col_sim_out = st.columns([1, 1.8])
    
    with col_sim_in:
        st.markdown("### ⚙️ Scenario Controls")
        sim_att = st.slider("Simulated Attendance", 100, 1200, 650, 25)
        sim_shift = st.selectbox("Shift", ["Lunch", "Dinner", "Breakfast"])
        sim_cat = st.selectbox("Category", ["Meat", "Vegetables", "Rice", "Soup"])
        sim_rain = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 5.0)
        sim_event = st.checkbox("Event in Progress", False)
        
        sim_manual_prep = st.number_input("Custom Kitchen Prep (Meals)", 100, 1500, 720, 10)

    # Compute simulation outputs
    sim_in_data = {
        "Meal": sim_shift,
        "Food_Category": sim_cat,
        "Canteen_Section": "A",
        "Day_of_Week": "Wednesday",
        "expected_attendance": sim_att,
        "temperature": 28.0,
        "rainfall": sim_rain,
        "holiday": 0,
        "special_event": int(sim_event),
        "previous_day_consumption": int(sim_att * 0.94),
        "Is_Weekend": 0,
        "Month": 7,
        "Day": 15
    }
    s_pred = predictor.predict_demand(sim_in_data)
    s_rec = optimize_meal_preparation(
        predicted_demand=s_pred["predicted_consumption"],
        expected_attendance=sim_att,
        food_category=sim_cat,
        special_event=int(sim_event),
        rainfall=sim_rain
    )
    
    surplus_manual = max(0, sim_manual_prep - s_pred["predicted_consumption"])
    surplus_opt = s_rec["expected_surplus"]
    
    waste_kg_man = round(surplus_manual * 0.44, 1)
    waste_kg_opt = round(surplus_opt * 0.44, 1)
    
    cost_man = round(waste_kg_man * 3.65, 2)
    cost_opt = round(waste_kg_opt * 3.65, 2)
    
    co2e_man = round(waste_kg_man * 2.5, 1)
    co2e_opt = round(waste_kg_opt * 2.5, 1)
    
    with col_sim_out:
        st.markdown("### 📊 Simulated Operational & Environmental Impact")
        
        k1, k2, k3 = st.columns(3)
        with k1:
            diff_w = waste_kg_man - waste_kg_opt
            st.metric("Waste Weight", f"{waste_kg_man} kg", delta=f"-{diff_w:.1f} kg with OptiMeal" if diff_w > 0 else "0.0 kg")
        with k2:
            diff_c = cost_man - cost_opt
            st.metric("Financial Cost", f"${cost_man}", delta=f"-${diff_c:.2f} savings" if diff_c > 0 else "$0.00")
        with k3:
            diff_co2 = co2e_man - co2e_opt
            st.metric("CO2e Footprint", f"{co2e_man} kg", delta=f"-{diff_co2:.1f} kg CO2e" if diff_co2 > 0 else "0.0 kg")
            
        chart_df = pd.DataFrame({
            "Metric": ["Waste Weight (kg)", "Financial Loss ($)", "Carbon Emissions (kg CO2e)"],
            "Manual Prep Strategy": [waste_kg_man, cost_man, co2e_man],
            "OptiMeal AI Strategy": [waste_kg_opt, cost_opt, co2e_opt]
        })
        
        fig_sim = px.bar(
            chart_df,
            x="Metric",
            y=["Manual Prep Strategy", "OptiMeal AI Strategy"],
            barmode="group",
            color_discrete_sequence=[COLOR_ORANGE, COLOR_EMERALD]
        )
        fig_sim = apply_custom_plotly_layout(fig_sim, "Strategy Comparison: Manual vs OptiMeal AI", height=310)
        st.plotly_chart(fig_sim, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: AUTOMATED AI INSIGHTS
# ---------------------------------------------------------
elif selected_page == "🧠 Automated AI Insights":
    st.markdown("## 🧠 Automated Operational Intelligence")
    st.markdown("Actionable findings generated directly from historical distributions and anomaly patterns.")
    
    insights = generate_rule_based_insights(df)
    
    for idx, ins in enumerate(insights):
        card_class = "insight-card-green" if idx % 3 == 0 else ("insight-card-orange" if idx % 3 == 1 else "insight-card-yellow")
        st.markdown(
            f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span class="card-title">{ins['title']}</span>
                    <span style="background-color: #FFFFFF; color: #1E293B; font-size: 0.72rem; font-weight: 700; padding: 3px 8px; border-radius: 6px; border: 1px solid #CBD5E1;">{ins['badge']}</span>
                </div>
                <div class="card-body">{ins['text']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# PAGE 6: SUSTAINABILITY ASSISTANT (RAG)
# ---------------------------------------------------------
elif selected_page == "📚 Sustainability Assistant (RAG)":
    st.markdown("## 📚 Sustainability Knowledge Assistant (RAG)")
    st.markdown("Ask practical questions on food recovery, EPA waste hierarchy, batch cooking, and responsible donation.")
    
    r_col1, r_col2 = st.columns([2, 1])
    
    with r_col2:
        st.markdown(
            """
            <div class="metric-card-yellow">
                <div class="metric-label">Verified Knowledge Base</div>
                <div style="font-size: 0.85rem; color: #78350F; margin-top: 6px; line-height: 1.5;">
                    • EPA Food Recovery Hierarchy<br>
                    • Institutional Kitchen Best Practices<br>
                    • Batch Cooking & JIT Protocols<br>
                    • UN SDG 12 & Climate Guidance
                </div>
                <div class="metric-subtext" style="color: #059669; margin-top: 8px;">✓ 100% Fact Grounded</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with r_col1:
        sample_q = st.selectbox(
            "Select a common sustainability query:",
            [
                "-- Select a question or type below --",
                "What is the EPA Food Recovery Hierarchy?",
                "What are common strategies to reduce food waste in cafeterias?",
                "What should organizations do with unavoidable food waste?",
                "How does food waste connect to SDG 12 and climate change?"
            ]
        )
        
        q_val = "" if sample_q.startswith("--") else sample_q
        user_query = st.text_input("Your question:", value=q_val)
        
        if st.button("Search Knowledge Base", use_container_width=True) and user_query:
            with st.spinner("Retrieving verified guidelines..."):
                rag_out = rag_assistant.query_knowledge_base(user_query)
                
                st.markdown("### 📖 Answer")
                st.markdown(
                    f"""
                    <div class="insight-card-green">
                        <div class="card-body">{rag_out['answer']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if rag_out["retrieved_sources"]:
                    st.markdown("#### 📄 Document Citations")
                    for s in rag_out["retrieved_sources"]:
                        with st.expander(f"Source: {s['source_file']} (Relevance: {s['relevance_score']}%)"):
                            st.write(s["content"])

