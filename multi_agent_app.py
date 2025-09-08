import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.data_processor import DataProcessor
from src.agents.sql_agent import NYC311SQLAgent
from src.agents.analysis_orchestrator import AnalysisOrchestrator
from src.utils.deepseek_client import DeepSeekClient
from config import DATABASE_PATH, CSV_PATH, DEEPSEEK_API_KEY

# Page config
st.set_page_config(
    page_title="NYC 311 Multi-Agent Analytics",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def initialize_multi_agent_system():
    """Initialize the multi-agent system with caching"""
    if not DEEPSEEK_API_KEY:
        st.error("Please set DEEPSEEK_API_KEY in your environment or .env file")
        st.stop()
    
    # Initialize DeepSeek client with both models
    deepseek_client = DeepSeekClient()
    
    # Initialize SQL agent
    sql_agent = NYC311SQLAgent(DATABASE_PATH, deepseek_client.client)
    
    # Initialize multi-agent orchestrator
    orchestrator = AnalysisOrchestrator(deepseek_client, sql_agent, DATABASE_PATH)
    
    return deepseek_client, sql_agent, orchestrator

def create_enhanced_chart(data, chart_type="bar", title=""):
    """Create enhanced chart with better styling"""
    if not data or len(data) == 0:
        return None
    
    df = pd.DataFrame(data)
    
    # Enhanced metric display
    if chart_type == "metric":
        if len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number",
                value=value,
                title={"text": title or df.columns[0].replace('_', ' ').title()},
                number={'font': {'size': 48}}
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            return fig
    
    # Enhanced charts with better styling
    elif chart_type == "bar":
        if len(df.columns) >= 2:
            x_col, y_col = df.columns[0], df.columns[1]
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=title or f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
                color=y_col,
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        else:
            fig = px.bar(df, title=title or "Data Visualization")
    
    elif chart_type == "pie":
        if len(df.columns) >= 2:
            names_col, values_col = df.columns[0], df.columns[1]
            fig = px.pie(
                df, names=names_col, values=values_col,
                title=title or f"Distribution of {names_col.replace('_', ' ').title()}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
        else:
            fig = px.pie(df, title=title or "Data Distribution")
    
    elif chart_type == "line":
        if len(df.columns) >= 2:
            x_col, y_col = df.columns[0], df.columns[1]
            fig = px.line(
                df, x=x_col, y=y_col,
                title=title or f"{y_col.replace('_', ' ').title()} Trend",
                labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
                markers=True
            )
        else:
            fig = px.line(df, title=title or "Trend Analysis")
    
    else:
        fig = px.bar(df, title=title or "Data Visualization")
    
    # Common styling improvements
    fig.update_layout(
        font=dict(size=12),
        title_font_size=16,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def display_execution_steps(steps):
    """Display the execution steps in an informative way"""
    st.subheader("🔍 Analysis Execution Steps")
    
    for i, step in enumerate(steps, 1):
        with st.expander(f"Step {i}: {step.get('step_id', 'Unknown')} ({'✅ Success' if step['success'] else '❌ Failed'})"):
            if step['success']:
                st.success(f"**Description:** {step.get('metadata', {}).get('step', {}).get('description', 'N/A')}")
                if 'row_count' in step.get('metadata', {}):
                    st.info(f"**Rows processed:** {step['metadata']['row_count']:,}")
                if 'query' in step.get('metadata', {}):
                    st.code(step['metadata']['query'], language='sql')
            else:
                st.error(f"**Error:** {step.get('error', 'Unknown error')}")

def main():
    st.title("🤖 NYC 311 Multi-Agent Analytics System")
    st.markdown("**Powered by DeepSeek Reasoner + Chat Models** | Advanced multi-step analysis with intelligent query planning")
    
    # Initialize multi-agent system
    with st.spinner("🚀 Initializing multi-agent system..."):
        deepseek_client, sql_agent, orchestrator = initialize_multi_agent_system()
    
    # Sidebar with system info
    with st.sidebar:
        st.header("🤖 Multi-Agent System")
        st.success("**Query Planner** (deepseek-reasoner)")
        st.success("**SQL Agent** (deepseek-chat)")
        st.success("**Analysis Orchestrator** (coordination)")
        
        st.header("📊 Quick Stats")
        processor = DataProcessor(CSV_PATH, DATABASE_PATH)
        summary = processor.get_data_summary()
        
        if summary:
            st.metric("Total Records", f"{summary.get('total_records', 0):,}")
            if 'date_range' in summary:
                st.write(f"**Date Range:** {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        st.header("💡 Multi-Step Examples")
        complex_questions = [
            "For the top 5 complaint types, what percent were closed within 3 days?",
            "Which Manhattan ZIP codes have the fastest resolution times?",
            "Compare noise complaint resolution across boroughs",
            "What's the seasonal trend for the top 3 complaint types?",
            "Show me the distribution of complaint types by borough"
        ]
        
        for question in complex_questions:
            if st.button(question, key=f"complex_{hash(question)}"):
                st.session_state.user_question = question
        
        st.header("🔧 Simple Questions") 
        simple_questions = [
            "What are the top 10 complaint types?",
            "Which borough has the most complaints?",
            "What percentage of complaints are closed within 3 days?",
            "What proportion of complaints include valid coordinates?"
        ]
        
        for question in simple_questions:
            if st.button(question, key=f"simple_{hash(question)}"):
                st.session_state.user_question = question
    
    # Main interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display chart if available
            if message.get("chart"):
                st.plotly_chart(message["chart"], use_container_width=True)
            
            # Display execution steps if available
            if message.get("execution_steps"):
                display_execution_steps(message["execution_steps"])
    
    # Handle sample question button clicks
    if st.session_state.get("user_question"):
        prompt = st.session_state.pop("user_question")
    else:
        prompt = st.chat_input("Ask a complex question about NYC 311 data...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response using multi-agent orchestrator
        with st.chat_message("assistant"):
            with st.spinner("🤖 Multi-agent system analyzing your question..."):
                try:
                    # Execute multi-agent analysis
                    result = orchestrator.execute_analysis(prompt)
                    
                    if result["success"]:
                        # Display natural language response
                        st.markdown(result["natural_language_response"])
                        
                        # Create and display chart
                        chart = None
                        if result["data"] and result["chart_type"] != "error":
                            chart = create_enhanced_chart(
                                result["data"], 
                                result["chart_type"],
                                title=f"Analysis Results for: {prompt[:50]}..."
                            )
                            
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                        
                        # Show raw data if available
                        if result["data"]:
                            with st.expander("📊 View Raw Data"):
                                st.dataframe(pd.DataFrame(result["data"]))
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["natural_language_response"],
                            "chart": chart,
                            "execution_steps": result.get("execution_steps", []),
                            "success": True
                        })
                        
                        # Show execution steps
                        if result.get("execution_steps"):
                            display_execution_steps(result["execution_steps"])
                        
                    else:
                        # Handle failure
                        error_msg = result["natural_language_response"]
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "execution_steps": result.get("execution_steps", []),
                            "success": False
                        })
                        
                        # Show failed steps for debugging
                        if result.get("execution_steps"):
                            display_execution_steps(result["execution_steps"])
                    
                except Exception as e:
                    error_msg = f"Multi-agent system error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "success": False
                    })
    
    # Footer
    st.markdown("---")
    st.markdown("**🤖 Multi-Agent Architecture:** Query Planner → Analysis Orchestrator → Specialized Agents → Coordinated Results")

if __name__ == "__main__":
    main()
