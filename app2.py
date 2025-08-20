import streamlit as st
import pandas as pd
import plotly.express as px

# ---- PAGE CONFIG ----
st.set_page_config(page_title="TechLance Benefits Dashboard", layout="wide")

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_data.csv")
    # Derived metrics
    df["Utilization"] = df["UsageFrequency"]
    df["Satisfaction"] = df["SatisfactionScore"]
    df["Benefit_Spend"] = df["BenefitCost"]
    # Simple ROI proxy = satisfaction per unit cost (scaled)
    df["ROI"] = (df["SatisfactionScore"] / (df["BenefitCost"] + 1)) * 100  
    return df

df = load_data()

# ---- SIDEBAR FILTERS ----
st.sidebar.header("Filters")
department = st.sidebar.multiselect("Department", df["Department"].dropna().unique())
age_group = st.sidebar.multiselect("Age Group", df["age_group"].dropna().unique())
benefit_type = st.sidebar.multiselect("Benefit SubType", df["BenefitSubType"].dropna().unique())

filtered = df.copy()
if department:
    filtered = filtered[filtered["Department"].isin(department)]
if age_group:
    filtered = filtered[filtered["age_group"].isin(age_group)]
if benefit_type:
    filtered = filtered[filtered["BenefitSubType"].isin(benefit_type)]

# ---- MAIN DASHBOARD ----
st.title("📊 TechLance Benefits Analytics Dashboard")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees", filtered["EmployeeID"].nunique())
col2.metric("Avg Utilization", round(filtered["Utilization"].mean(), 2))
col3.metric("Avg Satisfaction", round(filtered["Satisfaction"].mean(), 2))
col4.metric("ROI Proxy (%)", round(filtered["ROI"].mean(), 2))

st.divider()

# ---- VISUALS ----
st.subheader("Benefit Spend by Department")
fig1 = px.bar(
    filtered.groupby("Department")["Benefit_Spend"].sum().reset_index(),
    x="Department", y="Benefit_Spend", color="Department"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Satisfaction by Age Group")
fig2 = px.box(filtered, x="age_group", y="Satisfaction", color="age_group")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Utilization by Benefit SubType")
fig3 = px.bar(
    filtered.groupby("BenefitSubType")["Utilization"].mean().reset_index(),
    x="BenefitSubType", y="Utilization", color="BenefitSubType"
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("ROI Proxy by Department")
fig4 = px.bar(
    filtered.groupby("Department")["ROI"].mean().reset_index(),
    x="Department", y="ROI", color="Department"
)
st.plotly_chart(fig4, use_container_width=True)

# ---- INSIGHTS ----
st.subheader("💡 Generative Insights (Rule-based placeholder)")
top_comments = filtered.groupby("BenefitSubType")["Satisfaction"].mean().sort_values(ascending=False).head(3)
st.write("Top performing benefits (by satisfaction):")
st.table(top_comments)
