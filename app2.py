import streamlit as st
import pandas as pd

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Benefits Analytics Dashboard", layout="wide")

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data.csv")

df = load_data()

# ---- SIDEBAR FILTERS ----
st.sidebar.header("Filters")

department = st.sidebar.multiselect(
    "Select Department", options=df["Department"].dropna().unique()
)
age = st.sidebar.multiselect(
    "Select Age Group", options=df["Age"].dropna().unique()
)
benefit_type = st.sidebar.multiselect(
    "Select Benefit Subtype", options=df["BenefitSubType"].dropna().unique()
)

filtered_df = df.copy()
if department:
    filtered_df = filtered_df[filtered_df["Department"].isin(department)]
if age:
    filtered_df = filtered_df[filtered_df["Age"].isin(age)]
if benefit_type:
    filtered_df = filtered_df[filtered_df["BenefitSubType"].isin(benefit_type)]

# ---- MAIN DASHBOARD ----
st.title("📊 Benefits Analytics Dashboard")

# Summary stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Avg Utilization", f"{filtered_df['Utilization'].mean():.2f}")
with col2:
    st.metric("Avg Satisfaction", f"{filtered_df['Satisfaction'].mean():.2f}")
with col3:
    st.metric("ROI", f"{filtered_df['ROI'].mean():.2f}")

st.divider()

# ---- DATA PREVIEW ----
st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(20))

# ---- PLACEHOLDER: GEN AI INSIGHTS ----
st.subheader("💡 Generative Insights")
st.info(
    "This section can be powered by a language model to summarize employee feedback, "
    "highlight trends, or provide recommendations."
)
