import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.data_processor import DataProcessor
from src.agents.sql_agent import NYC311SQLAgent
from src.agents.analysis_orchestrator import AnalysisOrchestrator
from src.agents.query_planner import QueryPlanner
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
    """Initialize multi-agent system components with caching"""
    if not DEEPSEEK_API_KEY:
        st.error("Please set DEEPSEEK_API_KEY in your environment or .env file")
        st.stop()
    
    # Initialize DeepSeek client with both models
    deepseek_client = DeepSeekClient()
    
    # Initialize SQL agent (for backward compatibility and fallbacks)
    sql_agent = NYC311SQLAgent(DATABASE_PATH, deepseek_client.client)
    
    # Initialize multi-agent orchestrator (the main brain)
    orchestrator = AnalysisOrchestrator(deepseek_client, sql_agent, DATABASE_PATH)
    
    return deepseek_client, sql_agent, orchestrator

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_summary():
    """Get cached data summary"""
    try:
        processor = DataProcessor(CSV_PATH, DATABASE_PATH)
        return processor.get_data_summary()
    except Exception as e:
        st.error(f"Failed to load data summary: {e}")
        return None

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

def display_execution_steps(steps):
    """Display the multi-agent execution steps for transparency"""
    if not steps or len(steps) == 0:
        return
    
    st.subheader("🤖 Multi-Agent Analysis Steps")
    
    for i, step in enumerate(steps, 1):
        success_icon = "✅" if step['success'] else "❌"
        step_name = step.get('step_id', f'Step {i}')
        
        with st.expander(f"{success_icon} {step_name} ({'Success' if step['success'] else 'Failed'})"):
            if step['success']:
                metadata = step.get('metadata', {})
                
                # Show step description
                if 'step' in metadata and 'description' in metadata['step']:
                    st.info(f"**Description:** {metadata['step']['description']}")
                
                # Show data results
                if 'row_count' in metadata:
                    st.metric("Rows Processed", f"{metadata['row_count']:,}")
                
                # Show SQL query if available
                if 'query' in metadata:
                    st.code(metadata['query'], language='sql')
                
                # Show sample data
                if step['data'] and len(step['data']) > 0:
                    sample_data = step['data'][:3]  # First 3 rows
                    st.json(sample_data, expanded=False)
                    
            else:
                st.error(f"**Error:** {step.get('error', 'Unknown error')}")

def detect_question_complexity(question):
    """Detect if a question requires multi-step analysis"""
    complex_patterns = [
        r"top\s+\d+.*?(percent|percentage|%)",  # "top 5... what percent"
        r"(compare|versus|vs|difference).*(across|between)",  # comparative analysis
        r"(which|what).*?(manhattan|brooklyn|queens|bronx|staten).*(fastest|slowest|highest|lowest)",  # filtered ranking
        r"(trend|over time|monthly|seasonal|pattern)",  # time analysis
        r"for.*?(each|every).*?(calculate|show|what)",  # iterative analysis
    ]
    
    question_lower = question.lower()
    for pattern in complex_patterns:
        if re.search(pattern, question_lower, re.IGNORECASE):
            return True
    
    return False

def main():
    st.title("🤖 NYC 311 Multi-Agent Analytics System")
    st.markdown("**Powered by DeepSeek Reasoner + Chat Models** | Advanced multi-step analysis with intelligent query planning")
    
    # Initialize session state first
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_question" not in st.session_state:
        st.session_state.user_question = None
    
    # Initialize multi-agent system (this is fast)
    deepseek_client, sql_agent, orchestrator = initialize_components()
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 Multi-Agent System")
        st.success("**Query Planner** (deepseek-reasoner)")
        st.success("**SQL Agent** (deepseek-chat)")
        st.success("**Analysis Orchestrator** (coordination)")
        st.info("💡 Automatically detects simple vs complex questions")
        
        st.header("📊 Quick Stats")
        
        # Load stats asynchronously with loading state
        with st.spinner("Loading statistics..."):
            summary = get_cached_summary()
        
        if summary:
            st.metric("Total Records", f"{summary.get('total_records', 0):,}")
            
            if 'date_range' in summary:
                st.write(f"**Date Range:** {summary['date_range']['start']} to {summary['date_range']['end']}")
            
            if 'top_complaint_types' in summary:
                st.write("**Top Complaint Types:**")
                for complaint_type, count in summary['top_complaint_types'][:3]:
                    st.write(f"• {complaint_type}: {count:,}")
        else:
            st.warning("Unable to load statistics. Data may not be available.")
        
        st.header("🔧 Multi-Step Questions")
        complex_questions = [
            "For the top 5 complaint types, what percent were closed within 3 days?",
            "What are the average closure times by borough?",
            "Show the top 10 ZIP codes with the most complaints and their closure rates"
        ]
        
        for question in complex_questions:
            if st.button(question, key=f"complex_{hash(question)}"):
                st.session_state.user_question = question
        
        st.header("💡 Simple Questions") 
        simple_questions = [
            "What are the top 10 complaint types?",
            "Which borough has the most complaints?",
            "What percentage of complaints are closed within 3 days?",
            "Show me complaints by ZIP code"
        ]
        
        for question in simple_questions:
            if st.button(question, key=f"simple_{hash(question)}"):
                st.session_state.user_question = question
        
        # Chat controls
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ New Chat", help="Start a fresh conversation"):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                if "user_question" in st.session_state:
                    st.session_state.user_question = None
                st.rerun()
        
        with col2:
            if st.button("📋 Clear", help="Clear current conversation"):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
        
        # Show conversation stats
        if "messages" in st.session_state and st.session_state.messages:
            st.markdown(f"**Conversation:** {len(st.session_state.messages)} messages")
    
    # Main chat interface
    # Add welcome message if this is the first time
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🤖 Welcome to the **Multi-Agent Analytics System**! I use advanced AI planning to analyze NYC 311 data with both **simple** and **complex multi-step** queries.\n\n💡 **Try asking:**\n• Simple: \"What are the top complaint types?\"\n• Complex: \"For the top 5 complaint types, what percent were closed within 3 days?\"\n\nI'll automatically detect complexity and coordinate multiple AI agents for comprehensive analysis!"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display chart if available
            if "chart" in message and message["chart"]:
                # Use message index as unique key to avoid ID conflicts
                message_index = st.session_state.messages.index(message)
                st.plotly_chart(message["chart"], use_container_width=True, key=f"chart_{message_index}")
            
            # Display execution steps for assistant messages if available
            if message["role"] == "assistant" and message.get("execution_steps"):
                display_execution_steps(message["execution_steps"])
    
    # Handle sample question button clicks
    if st.session_state.get("user_question"):
        prompt = st.session_state.pop("user_question")
    else:
        # Chat input - this will always be available for continuous conversation
        prompt = st.chat_input("Ask a question about NYC 311 data...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response using multi-agent orchestrator
        with st.chat_message("assistant"):
            # Show analysis type detection
            is_complex = detect_question_complexity(prompt)
            analysis_type = "🤖 **Multi-Step Analysis**" if is_complex else "⚡ **Data** Analysis**"
            st.info(f"{analysis_type} - {('Planning multi-agent coordination...' if is_complex else 'Analysing...')}")
            
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
                            chart_type = result["chart_type"]
                            chart = create_chart(result["data"], chart_type)
                            
                            if chart:
                                # Use timestamp as unique key for new charts
                                chart_key = f"multi_chart_{int(time.time() * 1000)}"
                                st.plotly_chart(chart, use_container_width=True, key=chart_key)
                        
                        # Show raw data if available
                        if result["data"]:
                            with st.expander("📊 View Raw Data"):
                                table_key = f"multi_table_{int(time.time() * 1000)}"
                                st.dataframe(pd.DataFrame(result["data"]), key=table_key)
                        
                        # Show execution steps for transparency
                        if result.get("execution_steps") and len(result["execution_steps"]) > 0:
                            display_execution_steps(result["execution_steps"])
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["natural_language_response"],
                            "chart": chart,
                            "execution_steps": result.get("execution_steps", []),
                            "analysis_type": analysis_type
                        })
                        
                    else:
                        # Handle analysis failure
                        error_msg = result["natural_language_response"]
                        st.error(error_msg)
                        
                        # Show failed steps for debugging if available
                        if result.get("execution_steps"):
                            display_execution_steps(result["execution_steps"])
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "execution_steps": result.get("execution_steps", [])
                        })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
                    st.error(error_msg)
                    
                    # Add helpful suggestions
                    st.info("💡 **Troubleshooting tips:**\n"
                           "- Try rephrasing your question\n"
                           "- Check if the data contains the information you're looking for\n"
                           "- Use the sample questions as a reference")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
        
        # Rerun to refresh the chat interface and allow for next question
        st.rerun()
    
    # Footer
    st.markdown("---")
    # st.markdown("**🤖 Multi-Agent Architecture:** Query Planner → Analysis Orchestrator → Specialized Agents → Coordinated Results")

if __name__ == "__main__":
    main()
