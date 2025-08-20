"""
TechLance Solutions — Phase 4: Analytics Dashboard & Recommendations
Standalone Streamlit app (run with `streamlit run app.py` or `python app.py` to auto-spawn).

Input: ./data/cleaned_data.csv with columns:
EmployeeID, BenefitID, UsageFrequency, LastUsedDate, Age, Gender, Department, Tenure,
BenefitType, BenefitSubType, BenefitCost, SatisfactionScore, Comments

Features
- Caching (st.cache_data) for faster reloads
- Filters: Department, Age range, BenefitSubType
- Metrics: Utilization, Avg Satisfaction, Segments, ROI (proxy)
- Visualizations: usage trends, satisfaction by benefit, department heatmap, segments
- "Generative AI insights": auto-summarized feedback themes in chart tooltips
- Export (optional placeholder): HTML snapshot generation function (not wired to button by default)

Dependencies: streamlit, pandas, numpy, altair, plotly
"""

from __future__ import annotations
import os
import sys
import subprocess
from datetime import datetime
import re

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

# -----------------------------
# Page config MUST be first Streamlit call
# -----------------------------
st.set_page_config(page_title="TechLance Benefits Dashboard", layout="wide")

COMPANY = "TechLance Solutions"
ANNUAL_BENEFITS_COST = 12_000_000

# -----------------------------
# Data Loading & Preparation
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "cleaned_data.csv")
COMMON_STOP = set(
    "a an and are as at be by for from has have he her his i in is it its of on or that the their they to was were will with you your our we not very more less just".split()
)

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize columns (strip spaces)
    df.columns = df.columns.str.strip()

    # Types
    if "LastUsedDate" in df.columns:
        df["LastUsedDate"] = pd.to_datetime(df["LastUsedDate"], errors="coerce")
    for col in ["UsageFrequency", "BenefitCost", "SatisfactionScore", "Age", "Tenure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived fields
    if {"BenefitCost", "UsageFrequency"}.issubset(df.columns):
        df["Benefit_Spend"] = df["BenefitCost"].fillna(0) * df["UsageFrequency"].fillna(0)
    df["Utilized"] = (df.get("UsageFrequency", 0) > 0).astype(int)

    # Age & Tenure bands for segments
    if "Age" in df.columns:
        age_bins = [0, 24, 34, 44, 54, 64, 200]
        age_labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
        df["AgeBand"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, include_lowest=True)
    if "Tenure" in df.columns:
        t_bins = [-1, 1, 3, 5, 10, 50]
        t_labels = ["<1y", "1-3y", "3-5y", "5-10y", "10y+"]
        df["TenureBand"] = pd.cut(df["Tenure"], bins=t_bins, labels=t_labels)

    return df

@st.cache_data(show_spinner=False)
def tokenize_comments(series: pd.Series) -> list[str]:
    tokens = []
    for c in series.dropna().astype(str):
        # simple tokenization
        ws = re.findall(r"[A-Za-z]{4,}", c.lower())
        ws = [w for w in ws if w not in COMMON_STOP]
        tokens.extend(ws)
    return tokens

@st.cache_data(show_spinner=False)
def feedback_themes(df: pd.DataFrame, top_n: int = 6) -> list[str]:
    if "Comments" not in df.columns or df["Comments"].dropna().empty:
        return []
    toks = tokenize_comments(df["Comments"])
    if not toks:
        return []
    vc = pd.Series(toks).value_counts().head(top_n)
    return [f"{w} (≈{int(n)})" for w, n in vc.items()]

@st.cache_data(show_spinner=False)
def feedback_themes_by_subtype(df: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    """Return a small table mapping BenefitSubType -> comma-joined top tokens."""
    if "Comments" not in df.columns or "BenefitSubType" not in df.columns:
        return pd.DataFrame(columns=["BenefitSubType", "Themes"])
    rows = []
    for sub, g in df.groupby("BenefitSubType"):
        toks = tokenize_comments(g["Comments"]) if not g.empty else []
        if toks:
            vc = pd.Series(toks).value_counts().head(top_n)
            rows.append({"BenefitSubType": sub, "Themes": ", ".join(vc.index.tolist())})
        else:
            rows.append({"BenefitSubType": sub, "Themes": ""})
    return pd.DataFrame(rows)

# -----------------------------
# KPI helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def compute_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "utilization_rate": 0.0,
            "avg_satisfaction": np.nan,
            "active_population": 0,
            "roi_index": np.nan,
        }
    total_emp = df["EmployeeID"].nunique()
    utilized_emp = df.groupby("EmployeeID")["Utilized"].max().sum()
    utilization_rate = utilized_emp / total_emp if total_emp else 0.0

    avg_sat = df["SatisfactionScore"].mean(skipna=True)

    # ROI proxy: (avg usage per employee × avg satisfaction) / avg benefit cost
    avg_usage_per_emp = df.groupby("EmployeeID")["UsageFrequency"].sum().mean()
    avg_cost_per_benefit = df["BenefitCost"].replace(0, np.nan).mean(skipna=True)
    roi_index = (
        (avg_usage_per_emp or 0) * (avg_sat or 0) / avg_cost_per_benefit
        if avg_cost_per_benefit and pd.notna(avg_sat)
        else np.nan
    )

    return {
        "utilization_rate": utilization_rate,
        "avg_satisfaction": avg_sat,
        "active_population": int(utilized_emp),
        "roi_index": roi_index,
    }

# -----------------------------
# Sidebar Filters
# -----------------------------

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    f = df.copy()
    if f.empty:
        return f

    # Department filter
    if "Department" in f.columns:
        depts = ["All"] + sorted([d for d in f["Department"].dropna().unique().tolist()])
        dept = st.sidebar.selectbox("Department", depts, index=0)
        if dept != "All":
            f = f[f["Department"] == dept]

    # Age range
    if "Age" in f.columns and f["Age"].notna().any():
        a_min, a_max = int(np.nanmin(f["Age"])), int(np.nanmax(f["Age"]))
        a1, a2 = st.sidebar.slider("Age Range", a_min, a_max, (a_min, a_max))
        f = f[(f["Age"] >= a1) & (f["Age"] <= a2)]

    # BenefitSubType multi-select
    if "BenefitSubType" in f.columns:
        subs = sorted(f["BenefitSubType"].dropna().unique().tolist())
        sel = st.sidebar.multiselect("Benefit SubType", options=subs, default=subs[: min(6, len(subs))])
        if sel:
            f = f[f["BenefitSubType"].isin(sel)]

    # Date range
    if "LastUsedDate" in f.columns and f["LastUsedDate"].notna().any():
        d_min, d_max = f["LastUsedDate"].min(), f["LastUsedDate"].max()
        d1, d2 = st.sidebar.date_input("Usage Date Range", value=(d_min, d_max))
        try:
            f = f[(f["LastUsedDate"] >= pd.to_datetime(d1)) & (f["LastUsedDate"] <= pd.to_datetime(d2))]
        except Exception:
            pass

    return f

# -----------------------------
# Visual Components
# -----------------------------

def kpi_cards(k: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Utilization Rate", f"{k['utilization_rate']:.1%}", help="% of employees with any usage in the selected period.")
    c2.metric("Avg Satisfaction", "N/A" if pd.isna(k["avg_satisfaction"]) else f"{k['avg_satisfaction']:.2f} / 5")
    c3.metric("Active Employees", f"{k['active_population']:,}")
    c4.metric("ROI Index (proxy)", "N/A" if pd.isna(k["roi_index"]) else f"{k['roi_index']:.3f}", help="(Avg usage/employee × Avg satisfaction) / Avg benefit cost")


def usage_trend(df: pd.DataFrame):
    if df.empty or "LastUsedDate" not in df.columns:
        st.info("Usage trend requires LastUsedDate.")
        return
    tmp = df.copy()
    tmp["Month"] = tmp["LastUsedDate"].dt.to_period("M").dt.to_timestamp()
    g = tmp.groupby("Month", as_index=False)["UsageFrequency"].sum()
    fig = px.line(g, x="Month", y="UsageFrequency", title="Usage Over Time")
    st.plotly_chart(fig, use_container_width=True)


def satisfaction_by_subtype(df: pd.DataFrame, themes_df: pd.DataFrame):
    if df.empty or "SatisfactionScore" not in df.columns:
        st.info("Satisfaction chart requires SatisfactionScore.")
        return
    agg = df.groupby(["BenefitType", "BenefitSubType"], as_index=False)["SatisfactionScore"].mean()
    if not themes_df.empty:
        agg = agg.merge(themes_df, on="BenefitSubType", how="left")
    chart = alt.Chart(agg).mark_bar().encode(
        x=alt.X("SatisfactionScore:Q", title="Avg Satisfaction"),
        y=alt.Y("BenefitSubType:N", sort='-x', title="Benefit SubType"),
        color="BenefitType:N",
        tooltip=[
            alt.Tooltip("BenefitType:N", title="Type"),
            alt.Tooltip("BenefitSubType:N", title="SubType"),
            alt.Tooltip("SatisfactionScore:Q", title="Avg Sat", format=".2f"),
            alt.Tooltip("Themes:N", title="Feedback Themes")
        ]
    ).properties(title="Average Satisfaction by Benefit SubType (with feedback themes)")
    st.altair_chart(chart, use_container_width=True)


def dept_utilization_heatmap(df: pd.DataFrame):
    if df.empty or "Department" not in df.columns:
        st.info("Department heatmap requires Department.")
        return
    agg = df.groupby(["Department", "BenefitType"], as_index=False)["Utilized"].mean()
    pivot = agg.pivot(index="Department", columns="BenefitType", values="Utilized").fillna(0)
    heat_df = pivot.reset_index().melt(id_vars="Department", var_name="BenefitType", value_name="UtilizationRate")
    fig = px.density_heatmap(heat_df, x="BenefitType", y="Department", z="UtilizationRate", title="Utilization Rate by Department & Benefit Type")
    st.plotly_chart(fig, use_container_width=True)


def segment_bubbles(df: pd.DataFrame):
    if df.empty or "AgeBand" not in df.columns or "TenureBand" not in df.columns:
        st.info("Segments require AgeBand and TenureBand.")
        return
    agg = df.groupby(["AgeBand", "TenureBand"], as_index=False)["EmployeeID"].nunique()
    agg.rename(columns={"EmployeeID": "Employees"}, inplace=True)
    chart = alt.Chart(agg).mark_circle(size=220).encode(
        x=alt.X("AgeBand:N", title="Age Band"),
        y=alt.Y("TenureBand:N", title="Tenure Band"),
        size=alt.Size("Employees:Q", title="# Employees"),
        tooltip=["AgeBand", "TenureBand", "Employees"]
    ).properties(title="Population Segments (Age × Tenure)")
    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Main App
# -----------------------------

def main():
    st.title(f"📊 {COMPANY} Benefits Analytics Dashboard")
    st.caption("Phase 4: Utilization • Satisfaction • Segments • ROI (proxy) • AI-assisted insights")

    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at {DATA_PATH}")
        st.stop()

    df = load_data(DATA_PATH)

    # Filters
    fdf = apply_filters(df)

    # KPIs
    kpis = compute_kpis(fdf)
    kpi_cards(kpis)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        usage_trend(fdf)
        segment_bubbles(fdf)

    with c2:
        themes_df = feedback_themes_by_subtype(fdf)
        satisfaction_by_subtype(fdf, themes_df)
        dept_utilization_heatmap(fdf)

    st.markdown("---")
    st.subheader("Generative Insights (auto-summarized feedback themes)")
    bullets = feedback_themes(fdf)
    if bullets:
        for b in bullets:
            st.write("• ", b)
    else:
        st.info("No feedback comments available for summarization.")

    st.success("Dashboard loaded successfully from cleaned_data.csv 🚀")

# -----------------------------
# CLI convenience: allow `python app.py` to spawn Streamlit
# -----------------------------
if __name__ == "__main__":
    # If invoked by streamlit, just run main
    if any("streamlit" in a for a in sys.argv):
        main()
    else:
        try:
            cmd = [sys.executable, "-m", "streamlit", "run", __file__]
            print("Launching Streamlit server... If nothing happens, run: streamlit run app.py")
            subprocess.run(cmd, check=False)
        except Exception as e:
            print("Failed to launch Streamlit automatically. Please run: streamlit run app.py")
            print(e)
