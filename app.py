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
from src.utils.deepseek_client import DeepSeekClient
from config import DATABASE_PATH, CSV_PATH, DEEPSEEK_API_KEY

# Page config
st.set_page_config(
    page_title="NYC 311 Data Analytics Agent",
    page_icon="🏙️",
    layout="wide"
)

@st.cache_resource
def initialize_components():
    """Initialize components with caching"""
    if not DEEPSEEK_API_KEY:
        st.error("Please set DEEPSEEK_API_KEY in your environment or .env file")
        st.stop()
    
    # Initialize DeepSeek client
    deepseek_client = DeepSeekClient()
    
    # Initialize SQL agent
    sql_agent = NYC311SQLAgent(DATABASE_PATH, deepseek_client.client)
    
    return deepseek_client, sql_agent

def determine_chart_type(df, user_question):
    """Intelligently determine the best chart type based on data and question"""
    question_lower = user_question.lower()
    
    # Single value metrics
    if len(df) == 1 and len(df.columns) == 1:
        return "metric"
    
    # Time series detection
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'period' in col.lower()]
    if date_columns and len(df) > 3:
        return "line"
    
    # Percentage/proportion questions -> pie chart
    if any(word in question_lower for word in ['percent', 'percentage', 'proportion', 'distribution', 'share']):
        return "pie"
    
    # Top N or ranking questions -> bar chart
    if any(word in question_lower for word in ['top', 'highest', 'most', 'largest', 'ranking']):
        return "bar"
    
    # Comparison questions -> bar chart
    if any(word in question_lower for word in ['compare', 'versus', 'vs', 'difference']):
        return "bar"
    
    # Default based on data structure
    if len(df.columns) >= 2 and len(df) <= 15:
        # Check if second column looks like counts/numbers for bar chart
        second_col = df.columns[1]
        if df[second_col].dtype in ['int64', 'float64']:
            return "bar"
    
    # Default fallback
    return "bar"

def create_chart(data, chart_type="bar"):
    """Create appropriate chart based on data and type"""
    if not data or len(data) == 0:
        return None
    
    df = pd.DataFrame(data)
    
    # Handle metric display
    if chart_type == "metric":
        if len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            # Create a simple metric display using a bar chart with single value
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number",
                value=value,
                title={"text": df.columns[0].replace('_', ' ').title()}
            ))
            fig.update_layout(height=200)
            return fig
    
    # Handle regular charts
    elif chart_type == "bar":
        if len(df.columns) >= 2:
            # Ensure proper column names for display
            x_col, y_col = df.columns[0], df.columns[1]
            x_title = x_col.replace('_', ' ').title()
            y_title = y_col.replace('_', ' ').title()
            
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=f"{y_title} by {x_title}",
                labels={x_col: x_title, y_col: y_title}
            )
            fig.update_layout(xaxis_tickangle=-45)
        else:
            fig = px.bar(df, title="Data Visualization")
    
    elif chart_type == "pie":
        if len(df.columns) >= 2:
            names_col, values_col = df.columns[0], df.columns[1]
            fig = px.pie(
                df, names=names_col, values=values_col,
                title=f"Distribution of {names_col.replace('_', ' ').title()}"
            )
        else:
            fig = px.pie(df, title="Data Distribution")
    
    elif chart_type == "line":
        if len(df.columns) >= 2:
            x_col, y_col = df.columns[0], df.columns[1]
            x_title = x_col.replace('_', ' ').title()
            y_title = y_col.replace('_', ' ').title()
            
            fig = px.line(
                df, x=x_col, y=y_col,
                title=f"{y_title} Trend Over {x_title}",
                labels={x_col: x_title, y_col: y_title}
            )
        else:
            fig = px.line(df, title="Trend Analysis")
    
    else:
        # Default to bar chart
        fig = px.bar(df, title="Data Visualization")
    
    # Common styling
    fig.update_layout(
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    st.title("🏙️ NYC 311 Data Analytics Agent")
    st.markdown("Ask questions about NYC 311 service requests data using natural language!")
    
    # Initialize components
    with st.spinner("Initializing components..."):
        deepseek_client, sql_agent = initialize_components()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Quick Stats")
        
        # SQL queries are executed in the background
        # Check logs in sql_commands.log file for debugging if needed
        
        # Get basic stats
        processor = DataProcessor(CSV_PATH, DATABASE_PATH)
        summary = processor.get_data_summary()
        
        if summary:
            st.metric("Total Records", f"{summary.get('total_records', 0):,}")
            
            if 'date_range' in summary:
                st.write(f"**Date Range:** {summary['date_range']['start']} to {summary['date_range']['end']}")
            
            if 'top_complaint_types' in summary:
                st.write("**Top Complaint Types:**")
                for complaint_type, count in summary['top_complaint_types'][:3]:
                    st.write(f"• {complaint_type}: {count:,}")
        
        st.header("💡 Sample Questions")
        sample_questions = [
            "What are the top 10 complaint types?",
            "Which borough has the most complaints?",
            "What percentage of complaints are closed within 3 days?",
            "Show me complaints by ZIP code",
            "What's the average closure time by complaint type?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.user_question = question
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display chart if available
            if "chart" in message and message["chart"]:
                st.plotly_chart(message["chart"], use_container_width=True)
    
    # Handle sample question button clicks
    if st.session_state.get("user_question"):
        prompt = st.session_state.pop("user_question")
    else:
        # Chat input
        prompt = st.chat_input("Ask a question about NYC 311 data...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                try:
                    # Get response from SQL agent
                    response = sql_agent.run(prompt)
                    
                    # Parse structured JSON data for visualization
                    chart = None
                    clean_response = response
                    
                    # Extract JSON data using the wrapper pattern
                    json_match = re.search(r"\[\[DATA_JSON\]\](.+?)\[\[/DATA_JSON\]\]", response, re.DOTALL)
                    if json_match:
                        try:
                            # Parse the JSON data
                            json_str = json_match.group(1).strip()
                            data = json.loads(json_str)
                            
                            if data and isinstance(data, list):
                                # Convert to DataFrame for charting
                                df = pd.DataFrame(data)
                                
                                # Intelligent chart type selection
                                chart_type = determine_chart_type(df, prompt)
                                chart = create_chart(data, chart_type)
                                
                                # Clean the response text (remove JSON wrapper)
                                clean_response = re.sub(
                                    r"\[\[DATA_JSON\]\].+?\[\[/DATA_JSON\]\]", 
                                    "", response, flags=re.DOTALL
                                ).strip()
                                
                        except (json.JSONDecodeError, Exception) as e:
                            st.warning(f"Could not parse visualization data: {e}")
                    
                    # Display the clean natural language response
                    st.markdown(clean_response)
                    
                    # Display chart if available
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # Also show raw data table for transparency
                        if json_match:
                            with st.expander("📊 View Raw Data"):
                                try:
                                    data = json.loads(json_match.group(1).strip())
                                    if data:
                                        st.dataframe(pd.DataFrame(data))
                                except:
                                    st.text("Raw data unavailable")
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": clean_response,
                        "chart": chart
                    })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Footer
    st.markdown("---")
    st.markdown("Built with LangGraph, DeepSeek, and Streamlit")

if __name__ == "__main__":
    main()
