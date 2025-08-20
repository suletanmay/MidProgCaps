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
    df["ROI"] = (df["SatisfactionScore"] / (df["BenefitCost"] + 1)) * 100  
    return df

df = load_data()

# ---- MAIN DASHBOARD ----
st.title("📊 TechLance Benefits Analytics Dashboard")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees", df["EmployeeID"].nunique())
col2.metric("Avg Utilization", round(df["Utilization"].mean(), 2))
col3.metric("Avg Satisfaction", round(df["Satisfaction"].mean(), 2))
col4.metric("ROI Proxy (%)", round(df["ROI"].mean(), 2))

st.divider()

# Plotly chart config (adds toolbar: zoom, reset, save, fullscreen)
plotly_config = {
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
    "displayModeBar": True
}

# ---- VISUAL 1 ----
st.subheader("Benefit Spend by Department")
dep_filter = st.multiselect("Select Department(s):", df["Department"].dropna().unique())
filtered1 = df[df["Department"].isin(dep_filter)] if dep_filter else df

fig1 = px.bar(
    filtered1.groupby("Department")["Benefit_Spend"].sum().reset_index(),
    x="Department", y="Benefit_Spend", color="Department"
)
st.plotly_chart(fig1, use_container_width=True, config=plotly_config)

# ---- VISUAL 2 ----
st.subheader("Satisfaction by Age Group")
age_filter = st.multiselect("Select Age Group(s):", df["age_group"].dropna().unique())
filtered2 = df[df["age_group"].isin(age_filter)] if age_filter else df

fig2 = px.box(filtered2, x="age_group", y="Satisfaction", color="age_group")
st.plotly_chart(fig2, use_container_width=True, config=plotly_config)

# ---- VISUAL 3 ----
st.subheader("Utilization by Benefit SubType")
benefit_filter = st.multiselect("Select Benefit SubType(s):", df["BenefitSubType"].dropna().unique())
filtered3 = df[df["BenefitSubType"].isin(benefit_filter)] if benefit_filter else df

fig3 = px.bar(
    filtered3.groupby("BenefitSubType")["Utilization"].mean().reset_index(),
    x="BenefitSubType", y="Utilization", color="BenefitSubType"
)
st.plotly_chart(fig3, use_container_width=True, config=plotly_config)

# ---- VISUAL 4 ----
st.subheader("ROI Proxy by Department")
dep_filter2 = st.multiselect("Select Department(s) for ROI:", df["Department"].dropna().unique())
filtered4 = df[df["Department"].isin(dep_filter2)] if dep_filter2 else df

fig4 = px.bar(
    filtered4.groupby("Department")["ROI"].mean().reset_index(),
    x="Department", y="ROI", color="Department"
)
st.plotly_chart(fig4, use_container_width=True, config=plotly_config)

# ---- INSIGHTS ----
st.subheader("💡 Generative Insights (Rule-based placeholder)")
top_comments = df.groupby("BenefitSubType")["Satisfaction"].mean().sort_values(ascending=False).head(3)
st.write("Top performing benefits (by satisfaction):")
st.table(top_comments)
