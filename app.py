import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="🚀 Interactive EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main background and text */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #bb86fc !important;
        font-weight: bold !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #bb86fc;
        color: #000000;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #9b6fd9;
    }
    
    /* Cards */
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load Dataset
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Display Dataset Info
def show_basic_info(df):
    st.markdown("### 📊 Dataset Overview")
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", f"{df.shape[1]:,}")
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    
    st.markdown("#### Preview Data")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📏 Dataset Shape")
        st.write(f"**Rows:** {df.shape[0]:,}  |  **Columns:** {df.shape[1]:,}")
    
    with col2:
        st.markdown("#### ⚠️ Missing Values")
        missing_values = df.isnull().sum()
        if not missing_values.empty:
            fig = px.bar(
                x=missing_values.index,
                y=missing_values.values,
                title="Missing Values by Column",
                color=missing_values.values,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                plot_bgcolor="#1e1e1e",
                paper_bgcolor="#1e1e1e",
                font_color="#ffffff"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found! 🎉")

    with st.expander("📌 Column Data Types", expanded=True):
        st.dataframe(df.dtypes.to_frame('Data Type'), use_container_width=True)

# Summary Statistics
def show_statistics(df):
    st.markdown("### 📈 Summary Statistics")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Basic Stats", "📈 Detailed Analysis"])
    
    with tab1:
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        numeric_df = df.select_dtypes(include=['number'])
        for col in numeric_df.columns:
            st.markdown(f"#### {col}")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    df,
                    x=col,
                    title=f"Distribution of {col}",
                    color_discrete_sequence=["#bb86fc"]
                )
                fig.update_layout(
                    plot_bgcolor="#1e1e1e",
                    paper_bgcolor="#1e1e1e",
                    font_color="#ffffff"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    df,
                    y=col,
                    title=f"Box Plot of {col}",
                    color_discrete_sequence=["#bb86fc"]
                )
                fig.update_layout(
                    plot_bgcolor="#1e1e1e",
                    paper_bgcolor="#1e1e1e",
                    font_color="#ffffff"
                )
                st.plotly_chart(fig, use_container_width=True)

# Data Visualizations
def show_visualizations(df):
    st.markdown("###  Interactive Visualizations")
    
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        st.warning("No numerical data found for visualizations.")
    else:
        # Correlation Heatmap
        st.markdown("#### Correlation Heatmap")
        fig = px.imshow(
            numeric_df.corr(),
            title="Correlation Matrix",
            color_continuous_scale="RdBu"
        )
        fig.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e",
            font_color="#ffffff"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Missing Values Heatmap
        st.markdown("#### Missing Values Heatmap")
        fig = px.imshow(
            df.isnull(),
            title="Missing Values Heatmap",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e",
            font_color="#ffffff"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter Plot Matrix
        st.markdown("#### Scatter Plot Matrix")
        fig = px.scatter_matrix(
            numeric_df,
            title="Scatter Plot Matrix",
            color_discrete_sequence=["#bb86fc"]
        )
        fig.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e",
            font_color="#ffffff"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Violin Plot
        st.markdown("#### Violin Plot")
        fig = px.violin(
            numeric_df,
            title="Violin Plot",
            color_discrete_sequence=["#bb86fc"]
        )
        fig.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e",
            font_color="#ffffff"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Histograms
        st.markdown("#### Histograms")
        for col in numeric_df.columns:
            fig = px.histogram(
                df,
                x=col,
                title=f"Histogram of {col}",
                color_discrete_sequence=["#bb86fc"]
            )
            fig.update_layout(
                plot_bgcolor="#1e1e1e",
                paper_bgcolor="#1e1e1e",
                font_color="#ffffff"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Box Plots
        st.markdown("#### Box Plots")
        for col in numeric_df.columns:
            fig = px.box(
                df,
                y=col,
                title=f"Box Plot of {col}",
                color_discrete_sequence=["#bb86fc"]
            )
            fig.update_layout(
                plot_bgcolor="#1e1e1e",
                paper_bgcolor="#1e1e1e",
                font_color="#ffffff"
            )
            st.plotly_chart(fig, use_container_width=True)
            

# AI summarizer : Auto-generate human-readable dataset summaries using LLMs.
# This function uses OpenRouter API to generate a summary of the dataset.

import streamlit as st
from utils.ai_summary import get_ai_summary_openrouter

# 🔐 Your OpenRouter credentials (replace with your actual values)
API_KEY = "your_api_key"
REFERER = "https://openrouter.ai/api/v1"
TITLE = "Interactive EDA Dashboard"

def show_ai_summary(df):
    st.markdown("### 🤖AI-Powered Dataset Summary")
    st.markdown("""
    This summary is generated by an AI model that analyzed the first 10 rows of your dataset to provide structural insights, trends, and suggestions.
    """)

    if st.button("🪄 Generate AI Summary"):
        with st.spinner("AI is analyzing your data..."):
            try:
                summary = get_ai_summary_openrouter(df, API_KEY, REFERER, TITLE)
                st.markdown("#### 🤖 Summary Output")
                st.write(summary)
            except Exception as e:
                st.error(f"Something went wrong: {e}")



# Main App
def main():
    st.title("🚀 Interactive EDA Dashboard")
    st.markdown("""
        <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #bb86fc;'>Welcome to the Interactive EDA Dashboard!</h3>
            <p>Upload your CSV file to explore and analyze your dataset with interactive visualizations.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            with st.sidebar:
                st.markdown("### 📌 Navigation")
                st.markdown("---")
                choice = st.radio(
                    "Choose Analysis Type:",
                    ["Basic Info", "Statistics", "Visualizations","AI Summary"],
                    label_visibility="collapsed"
                )
                st.markdown("---")
                st.markdown("### 📊 Dataset Info")
                st.markdown(f"**Rows:** {df.shape[0]:,}")
                st.markdown(f"**Columns:** {df.shape[1]:,}")
                st.markdown(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            if choice == "Basic Info":
                show_basic_info(df)
            elif choice == "Statistics":
                show_statistics(df)
            elif choice == "Visualizations":
                show_visualizations(df)
            elif choice == "AI Summary":
                show_ai_summary(df)

            # Additional Features
            st.markdown("### 🤔 Additional Features")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Generate Summary Report"):
                    st.markdown("### 📊 Summary Report")
                    st.write(df.describe())
            with col2:
                if st.button("📈 Generate Correlation Matrix"):
                    st.markdown("### 📈 Correlation Matrix")
                    numeric_df = df.select_dtypes(include=['number'])
                    fig = px.imshow(
                        numeric_df.corr(),
                        title="Correlation Matrix",
                        color_continuous_scale="RdBu"
                    )
                    fig.update_layout(
                        plot_bgcolor="#1e1e1e",
                        paper_bgcolor="#1e1e1e",
                        font_color="#ffffff"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Interactive Visualizations
            st.markdown("### 📊 Interactive Visualizations")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📈 Generate Histogram"):
                    st.markdown("### 📈 Histogram")
                    numeric_df = df.select_dtypes(include=['number'])
                    fig = px.histogram(
                        df,
                        x=numeric_df.columns[0],
                        title=f"Histogram of {numeric_df.columns[0]}",
                        color_discrete_sequence=["#bb86fc"]
                    )
                    fig.update_layout(
                        plot_bgcolor="#1e1e1e",
                        paper_bgcolor="#1e1e1e",
                        font_color="#ffffff"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if st.button("📈 Generate Box Plot"):
                    st.markdown("### 📈 Box Plot")
                    numeric_df = df.select_dtypes(include=['number'])
                    fig = px.box(
                        df,
                        y=numeric_df.columns[0],
                        title=f"Box Plot of {numeric_df.columns[0]}",
                        color_discrete_sequence=["#bb86fc"]
                    )
                    fig.update_layout(
                        plot_bgcolor="#1e1e1e",
                        paper_bgcolor="#1e1e1e",
                        font_color="#ffffff"
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
