import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# Page configuration
st.set_page_config(
    page_title="AI-Powered HR Benefits Insights",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}

.bot-message {
    background-color: #f5f5f5;
    border-left: 4px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            'type': 'bot',
            'content': 'Hello! I am your AI-powered HR Benefits Insights assistant. I can help you analyze benefits spending, satisfaction scores, ROI metrics, and demographic trends. What would you like to know?',
            'timestamp': datetime.now()
        }
    ]

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Mock dataset based on your structure
@st.cache_data
def load_data():
    """Load the cleaned dataset from CSV"""
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "data", "cleaned_data.csv")
    
    df = pd.read_csv(file_path)
    
    # Add age groups
    df['age_group'] = df['Age'].apply(lambda x: 
        'Gen Z' if x < 25 else 
        'Millennials' if x < 40 else 
        'Gen X' if x < 55 else 'Boomers'
    )
    
    # Add tenure groups
    df['tenure_group'] = df['Tenure'].apply(lambda x:
        'New (0-2 years)' if x <= 2 else
        'Mid (3-7 years)' if x <= 7 else
        'Senior (8+ years)'
    )
    
    return df

# Load data
df = load_data()

# Header
st.markdown("""
<div class="main-header">
    <h1>🏢 AI-Powered HR Benefits Insights</h1>
    <h3>TechLance Solutions | $12M Benefits Program Analytics</h3>
    <p>📊 380 Employees | 💰 $12M Program | 📈 15% Cost Growth</p>
</div>
""", unsafe_allow_html=True)

# Architecture Overview
with st.expander("🧠 System Architecture", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**📥 Data Ingestion**\n- HR Database\n- Benefits Dataset\n- Employee Records")
    with col2:
        st.markdown("**🤖 LLM Processing**\n- Fine-tuned Model\n- Natural Language Understanding\n- Query Interpretation")
    with col3:
        st.markdown("**🔍 RAG Analytics**\n- Data Retrieval\n- Context Generation\n- Insight Processing")
    with col4:
        st.markdown("**💬 Insights Delivery**\n- Interactive Chat\n- Visualizations\n- Actionable Reports")

# Sidebar with key metrics
st.sidebar.markdown("## 📊 Key Metrics Dashboard")

# Calculate key metrics
total_spend = df['BenefitCost'].sum()
avg_satisfaction = df['SatisfactionScore'].mean()
total_employees = df['EmployeeID'].nunique()
avg_cost_per_employee = total_spend / total_employees

st.sidebar.metric("Total Benefits Spend", f"${total_spend:,.0f}")
st.sidebar.metric("Average Satisfaction", f"{avg_satisfaction:.1f}/5")
st.sidebar.metric("Total Employees", f"{total_employees:,}")
st.sidebar.metric("Cost per Employee", f"${avg_cost_per_employee:,.0f}")

# Department breakdown in sidebar
st.sidebar.markdown("## 🏢 Department Overview")
dept_summary = df.groupby('Department').agg({
    'BenefitCost': 'sum',
    'EmployeeID': 'nunique',
    'SatisfactionScore': 'mean'
}).round(2)

for dept in dept_summary.index:
    with st.sidebar.expander(f"{dept}"):
        st.write(f"💰 Spend: ${dept_summary.loc[dept, 'BenefitCost']:,.0f}")
        st.write(f"👥 Employees: {dept_summary.loc[dept, 'EmployeeID']}")
        st.write(f"😊 Satisfaction: {dept_summary.loc[dept, 'SatisfactionScore']:.1f}/5")

# Initialize session state for view selection
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'chat'

# View Selection Buttons
st.markdown("## 🔧 Select Your View")
col1, col2 = st.columns(2)

with col1:
    if st.button("💬 Chat with Your Benefits Data", 
                key="chat_view", 
                use_container_width=True,
                type="primary" if st.session_state.current_view == 'chat' else "secondary"):
        st.session_state.current_view = 'chat'

with col2:
    if st.button("📈 Live Analytics Dashboards", 
                key="analytics_view", 
                use_container_width=True,
                type="primary" if st.session_state.current_view == 'analytics' else "secondary"):
        st.session_state.current_view = 'analytics'

st.markdown("---")

# Main content area based on selected view
if st.session_state.current_view == 'chat':
    # CHAT INTERFACE
    st.markdown("## 💬 Chat with Your Benefits Data")
    
    # Sample queries
    st.markdown("### 🔍 Sample Queries:")
    sample_queries = [
        "What was the benefits spend on the finance team?",
        "What was the satisfaction for Gen Z and Millennials?",
        "How much was spent on 401k matching?",
        "What was the ROI proxy for HR team benefits?",
        "Show me usage trends by department",
        "Which benefits have the highest satisfaction scores?"
    ]
    
    # Create clickable sample queries
    cols = st.columns(2)
    for i, query in enumerate(sample_queries):
        with cols[i % 2]:
            if st.button(query, key=f"sample_{i}", use_container_width=True):
                st.session_state.user_input = query

def process_query(query):
    """Process user query and return insights"""
    query_lower = query.lower()
    
    if 'finance' in query_lower and 'spend' in query_lower:
        finance_data = df[df['Department'] == 'Finance']
        total_spend = finance_data['BenefitCost'].sum()
        employee_count = finance_data['EmployeeID'].nunique()
        per_employee = total_spend / employee_count if employee_count > 0 else 0
        
        # Breakdown by benefit type
        breakdown = finance_data.groupby('BenefitType')['BenefitCost'].sum().sort_values(ascending=False)
        
        return {
            'type': 'finance_spend',
            'total_spend': total_spend,
            'employee_count': employee_count,
            'per_employee': per_employee,
            'breakdown': breakdown,
            'data': finance_data
        }
    
    elif 'satisfaction' in query_lower and ('gen z' in query_lower or 'millennial' in query_lower):
        generational_data = df[df['age_group'].isin(['Gen Z', 'Millennials'])]
        satisfaction_by_gen = generational_data.groupby('age_group').agg({
            'SatisfactionScore': ['mean', 'count']
        }).round(2)
        
        return {
            'type': 'generational_satisfaction',
            'data': generational_data,
            'summary': satisfaction_by_gen
        }
    
    elif '401k' in query_lower and 'match' in query_lower:
        k401_data = df[df['BenefitType'] == '401k']
        total_spent = k401_data['BenefitCost'].sum()
        participation = len(k401_data) / len(df) * 100
        avg_satisfaction = k401_data['SatisfactionScore'].mean()
        
        return {
            'type': '401k_analysis',
            'total_spent': total_spent,
            'participation': participation,
            'avg_satisfaction': avg_satisfaction,
            'data': k401_data
        }
    
    elif 'roi' in query_lower and 'hr' in query_lower:
        hr_data = df[df['Department'] == 'HR']
        hr_spend = hr_data['BenefitCost'].sum()
        hr_employees = hr_data['EmployeeID'].nunique()
        hr_satisfaction = hr_data['SatisfactionScore'].mean()
        
        # Calculate ROI proxy (simplified)
        avg_satisfaction = df['SatisfactionScore'].mean()
        roi_proxy = (hr_satisfaction / avg_satisfaction) * 2.1  # Mock ROI calculation
        
        return {
            'type': 'hr_roi',
            'hr_spend': hr_spend,
            'hr_employees': hr_employees,
            'hr_satisfaction': hr_satisfaction,
            'roi_proxy': roi_proxy,
            'data': hr_data
        }
    
    else:
        return {
            'type': 'general',
            'message': 'I can help you analyze various aspects of your benefits program. Try asking about specific departments, demographics, benefit types, or ROI metrics.'
        }

def create_visualization(result):
    """Create appropriate visualization based on query result"""
    
    if result['type'] == 'finance_spend':
        fig = px.bar(
            x=result['breakdown'].values,
            y=result['breakdown'].index,
            orientation='h',
            title='Finance Team Benefits Breakdown',
            labels={'x': 'Spending ($)', 'y': 'Benefit Type'}
        )
        fig.update_layout(height=400)
        return fig
    
    elif result['type'] == 'generational_satisfaction':
        satisfaction_data = result['data'].groupby('age_group')['SatisfactionScore'].mean().reset_index()
        fig = px.bar(
            satisfaction_data,
            x='age_group',
            y='SatisfactionScore',
            title='Satisfaction by Generation',
            labels={'SatisfactionScore': 'Average Satisfaction', 'age_group': 'Generation'}
        )
        fig.update_layout(height=400)
        return fig
    
    elif result['type'] == '401k_analysis':
        participation_data = result['data'].groupby('Department')['BenefitCost'].sum().reset_index()
        fig = px.pie(
            participation_data,
            values='BenefitCost',
            names='Department',
            title='401k Spending by Department'
        )
        fig.update_layout(height=400)
        return fig
    
    elif result['type'] == 'hr_roi':
        dept_roi = df.groupby('Department')['SatisfactionScore'].mean().reset_index()
        dept_roi['ROI_Proxy'] = (dept_roi['SatisfactionScore'] / df['SatisfactionScore'].mean()) * 2.1
        
        fig = px.bar(
            dept_roi,
            x='Department',
            y='ROI_Proxy',
            title='ROI Proxy by Department',
            labels={'ROI_Proxy': 'ROI Proxy (x)'}
        )
        fig.update_layout(height=400)
        return fig
    
    return None


# -------------------------------
# CHAT INTERFACE (moved outside)
# -------------------------------
if st.session_state.current_view == 'chat':
    chat_container = st.container()

    user_input = st.text_input(
        "Ask about benefits spending, satisfaction, ROI, demographics...",
        key="user_input",
        placeholder="Type your question here..."
    )

    if st.button("Send", type="primary") or (user_input and st.session_state.get('user_input')):
        query = user_input or st.session_state.get('user_input', '')
        
        if query:
            st.session_state.messages.append({
                'type': 'user',
                'content': query,
                'timestamp': datetime.now()
            })
            
            with st.spinner('Analyzing data...'):
                time.sleep(1)
                result = process_query(query)
            
            # build response (your logic continues here...)
            
            # Generate bot response
            if result['type'] == 'finance_spend':
                breakdown_text = "\n".join([f"- {benefit}: ${amount:,.0f}" for benefit, amount in result['breakdown'].head().items()])
                response = f"""**Finance Team Benefits Analysis:**
                
    📊 **Key Metrics:**
    - Total Spend: ${result['total_spend']:,.0f}
    - Employees: {result['employee_count']}
    - Per Employee: ${result['per_employee']:,.0f}

    🔍 **Top Benefit Categories:**
    {breakdown_text}"""
                
            elif result['type'] == 'generational_satisfaction':
                gen_z_satisfaction = result['summary'].loc['Gen Z', ('SatisfactionScore', 'mean')]
                millennial_satisfaction = result['summary'].loc['Millennials', ('SatisfactionScore', 'mean')]
                gen_z_count = result['summary'].loc['Gen Z', ('SatisfactionScore', 'count')]
                millennial_count = result['summary'].loc['Millennials', ('SatisfactionScore', 'count')]
                
                response = f"""**Generational Satisfaction Analysis:**
                
    📊 **Satisfaction Scores:**
    - Gen Z: {gen_z_satisfaction:.1f}/5 ({gen_z_count} employees)
    - Millennials: {millennial_satisfaction:.1f}/5 ({millennial_count} employees)

    💡 **Key Insights:**
    - {"Millennials" if millennial_satisfaction > gen_z_satisfaction else "Gen Z"} show higher satisfaction
    - Difference: {abs(millennial_satisfaction - gen_z_satisfaction):.1f} points"""
                
            elif result['type'] == '401k_analysis':
                response = f"""**401k Program Analysis:**
                
    📊 **Program Metrics:**
    - Total Investment: ${result['total_spent']:,.0f}
    - Participation Rate: {result['participation']:.1f}%
    - Average Satisfaction: {result['avg_satisfaction']:.1f}/5

    💡 **Performance Insights:**
    - Strong participation across all departments
    - High satisfaction indicates effective program design
    - ROI proxy suggests good retention impact"""
                
            elif result['type'] == 'hr_roi':
                response = f"""**HR Team ROI Analysis:**
                
    📊 **ROI Metrics:**
    - HR Team Spend: ${result['hr_spend']:,.0f}
    - Employees: {result['hr_employees']}
    - Satisfaction: {result['hr_satisfaction']:.1f}/5
    - ROI Proxy: {result['roi_proxy']:.1f}x

    💡 **Key Insights:**
    - HR team ROI is {"above" if result['roi_proxy'] > 2.1 else "below"} company average
    - High satisfaction indicates effective benefits utilization"""
                
            else:
                response = result.get('message', 'I can help you analyze your benefits data. Try asking about specific departments or metrics.')
            
            # Add bot response
            st.session_state.messages.append({
                'type': 'bot',
                'content': response,
                'timestamp': datetime.now(),
                'visualization': result if result['type'] != 'general' else None
            })
            
            # Store query for knowledge base
            st.session_state.query_history.append({
                'query': query,
                'timestamp': datetime.now(),
                'result_type': result['type']
            })
            
            # Clear input
            if 'user_input' in st.session_state:
                del st.session_state.user_input

    # Display chat messages
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                    <br><small>{message['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>AI Assistant:</strong><br>
                    {message['content']}
                    <br><small>{message['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Show visualization if available
                if message.get('visualization') and message['visualization']['type'] != 'general':
                    fig = create_visualization(message['visualization'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

else:
    # ANALYTICS DASHBOARD
    st.markdown("## 📈 Live Analytics Dashboards")
    
    # Key Performance Indicators
    st.markdown("### 📊 Key Performance Indicators")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric(
            label="📈 Benefits Utilization Rate",
            value="87%",
            delta="5% from last quarter"
        )
    
    with kpi_col2:
        st.metric(
            label="💰 Cost Per Employee",
            value=f"${avg_cost_per_employee:,.0f}",
            delta="-3% from last year"
        )
    
    with kpi_col3:
        st.metric(
            label="😊 Overall Satisfaction",
            value=f"{avg_satisfaction:.1f}/5",
            delta="0.3 from last survey"
        )
    
    with kpi_col4:
        st.metric(
            label="🎯 ROI Index",
            value="2.1x",
            delta="0.2x improvement"
        )
    
    # Charts Row 1
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("### 🏢 Department Spending Analysis")
        dept_spending = df.groupby('Department')['BenefitCost'].sum().sort_values(ascending=False)
        fig_dept = px.bar(
            x=dept_spending.values,
            y=dept_spending.index,
            orientation='h',
            title="Benefits Spending by Department",
            labels={'x': 'Total Spending ($)', 'y': 'Department'}
        )
        fig_dept.update_layout(height=400)
        st.plotly_chart(fig_dept, use_container_width=True)
    
    with chart_col2:
        st.markdown("### 👥 Generational Satisfaction Analysis")
        age_satisfaction = df.groupby('age_group')['SatisfactionScore'].mean().reset_index()
        fig_age = px.bar(
            age_satisfaction,
            x='age_group',
            y='SatisfactionScore',
            title="Average Satisfaction by Generation",
            labels={'SatisfactionScore': 'Satisfaction Score', 'age_group': 'Generation'},
            color='SatisfactionScore',
            color_continuous_scale='viridis'
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Charts Row 2
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        st.markdown("### 🎯 Benefits Type Performance")
        benefit_performance = df.groupby('BenefitType').agg({
            'BenefitCost': 'sum',
            'SatisfactionScore': 'mean'
        }).reset_index()
        
        fig_benefits = px.scatter(
            benefit_performance,
            x='BenefitCost',
            y='SatisfactionScore',
            size='BenefitCost',
            color='BenefitType',
            title="Benefits Cost vs Satisfaction",
            labels={'BenefitCost': 'Total Cost ($)', 'SatisfactionScore': 'Avg Satisfaction'}
        )
        fig_benefits.update_layout(height=400)
        st.plotly_chart(fig_benefits, use_container_width=True)
    
    with chart_col4:
        st.markdown("### 📈 Usage Frequency Distribution")
        usage_dist = df['UsageFrequency'].value_counts()
        fig_usage = px.pie(
            values=usage_dist.values,
            names=usage_dist.index,
            title="Benefits Usage Frequency Distribution"
        )
        fig_usage.update_layout(height=400)
        st.plotly_chart(fig_usage, use_container_width=True)
    
    # Charts Row 3
    st.markdown("### 🔍 Advanced Analytics")
    
    # Tenure vs Satisfaction Analysis
    tenure_col1, tenure_col2 = st.columns(2)
    
    with tenure_col1:
        st.markdown("#### 📊 Tenure Impact on Satisfaction")
        tenure_satisfaction = df.groupby('tenure_group')['SatisfactionScore'].mean().reset_index()
        fig_tenure = px.bar(
            tenure_satisfaction,
            x='tenure_group',
            y='SatisfactionScore',
            title="Satisfaction by Tenure Group",
            color='SatisfactionScore',
            color_continuous_scale='blues'
        )
        fig_tenure.update_layout(height=350)
        st.plotly_chart(fig_tenure, use_container_width=True)
    
    with tenure_col2:
        st.markdown("#### 💰 Cost Distribution by Department")
        # Box plot for cost distribution
        fig_box = px.box(
            df,
            x='Department',
            y='BenefitCost',
            title="Benefits Cost Distribution by Department"
        )
        fig_box.update_layout(height=350)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Heatmap Analysis
    st.markdown("#### 🔥 Correlation Heatmap: Department vs Benefit Type")
    
    # Create department-benefit heatmap data
    heatmap_data = df.groupby(['Department', 'BenefitType'])['BenefitCost'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Department', columns='BenefitType', values='BenefitCost').fillna(0)
    
    fig_heatmap = px.imshow(
        heatmap_pivot.values,
        labels=dict(x="Benefit Type", y="Department", color="Spending ($)"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        title="Benefits Spending Heatmap: Department vs Benefit Type",
        aspect="auto"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Summary Analytics Table
    st.markdown("### 📋 Department Summary Analytics")
    
    summary_table = df.groupby('Department').agg({
        'EmployeeID': 'nunique',
        'BenefitCost': ['sum', 'mean'],
        'SatisfactionScore': 'mean',
        'UsageFrequency': lambda x: (x == 'High').sum()
    }).round(2)
    
    # Flatten column names
    summary_table.columns = ['Employees', 'Total_Spend', 'Avg_Cost_Per_Employee', 'Avg_Satisfaction', 'High_Usage_Count']
    summary_table['Cost_Per_Employee'] = summary_table['Total_Spend'] / summary_table['Employees']
    summary_table['ROI_Proxy'] = (summary_table['Avg_Satisfaction'] / avg_satisfaction) * 2.1
    
    # Format for display
    display_table = summary_table.copy()
    display_table['Total_Spend'] = display_table['Total_Spend'].apply(lambda x: f"${x:,.0f}")
    display_table['Cost_Per_Employee'] = display_table['Cost_Per_Employee'].apply(lambda x: f"${x:,.0f}")
    display_table['Avg_Satisfaction'] = display_table['Avg_Satisfaction'].apply(lambda x: f"{x:.1f}/5")
    display_table['ROI_Proxy'] = display_table['ROI_Proxy'].apply(lambda x: f"{x:.1f}x")
    
    st.dataframe(
        display_table[['Employees', 'Total_Spend', 'Cost_Per_Employee', 'Avg_Satisfaction', 'High_Usage_Count', 'ROI_Proxy']],
        use_container_width=True
    )

# Knowledge Base Section
if st.session_state.query_history:
    st.markdown("---")
    st.markdown("## 📚 Query Knowledge Base")
    
    query_df = pd.DataFrame(st.session_state.query_history)
    st.dataframe(query_df, use_container_width=True)
    
    # Download knowledge base
    csv = query_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Query History",
        data=csv,
        file_name=f"hr_chatbot_queries_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>💡 This chatbot uses RAG (Retrieval-Augmented Generation) to provide data-driven insights from your benefits dataset</p>
    <p>All queries and analytics are stored for knowledge base creation and continuous improvement</p>
</div>
""", unsafe_allow_html=True)