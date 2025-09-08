from typing import Literal, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.tools import tool
import pandas as pd
import logging
import json
import sqlite3
from datetime import datetime

# Configure logging for SQL commands
sql_logger = logging.getLogger('sql_commands')
sql_logger.setLevel(logging.INFO)

# Create file handler for SQL logging
sql_handler = logging.FileHandler('sql_commands.log')
sql_handler.setLevel(logging.INFO)

# Create formatter
sql_formatter = logging.Formatter('%(asctime)s - %(message)s')
sql_handler.setFormatter(sql_formatter)

# Add handler to logger
sql_logger.addHandler(sql_handler)

logger = logging.getLogger(__name__)

class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], "messages"]

class NYC311SQLAgent:
    def __init__(self, db_path: str, llm):
        self.db_path = db_path
        self.llm = llm
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
        self.tools = self.toolkit.get_tools()
        self.tool_node = ToolNode(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        builder = StateGraph(MessagesState)
        
        # Add nodes
        builder.add_node("list_tables", self._list_tables)
        builder.add_node("get_schema", self._get_schema)
        builder.add_node("generate_query", self._generate_query)
        builder.add_node("check_query", self._check_query)
        builder.add_node("execute_query", self._execute_query)
        builder.add_node("format_response", self._format_response)
        
        # Add edges
        builder.add_edge(START, "list_tables")
        builder.add_edge("list_tables", "get_schema")
        builder.add_edge("get_schema", "generate_query")
        builder.add_conditional_edges(
            "generate_query",
            self._should_continue,
            {
                "check_query": "check_query",
                "format_response": "format_response"
            }
        )
        builder.add_edge("check_query", "execute_query")
        builder.add_edge("execute_query", "format_response")
        
        return builder.compile()
    
    def _list_tables(self, state: MessagesState):
        """List available tables"""
        tables = self.db.get_usable_table_names()
        response = f"Available tables: {', '.join(tables)}"
        return {"messages": [AIMessage(content=response)]}
    
    def _get_schema(self, state: MessagesState):
        """Get schema for relevant tables"""
        # For NYC 311, we know the main table is 'complaints'
        schema = self.db.get_table_info(["complaints"])
        response = f"Schema for complaints table:\n{schema}"
        return {"messages": [AIMessage(content=response)]}
    
    def _generate_query(self, state: MessagesState):
        """Generate SQL query using LLM with forced tool usage"""
        system_prompt = """
        You are a SQL expert for NYC 311 complaints data.Dont generate fake data or facts.Get data from database by runnig sql queries.
        You MUST use the sql_db_query tool to execute SQL queries and answer the user's question.
        
        Available columns: unique_key, created_date, closed_date, complaint_type, 
        descriptor, status, incident_zip, borough, latitude, longitude, agency and others
        
        Rules:
        - You MUST use the sql_db_query tool to execute SQL queries
        - Only use SELECT statements
        - Limit results to top 10 unless specified
        - Use proper SQLite syntax
        - Handle NULL values appropriately
        - Do not provide conversational responses - always execute SQL queries
        
        IMPORTANT: Use the sql_db_query tool to answer the user's question with actual data.
        """
        
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        
        # Log the user's question
        user_question = state["messages"][-1].content if state["messages"] else "Unknown"
        sql_logger.info(f"USER QUESTION: {user_question}")
        
        # Force the LLM to use SQL tools
        llm_with_tools = self.llm.bind_tools(
            [QuerySQLDataBaseTool(db=self.db)], 
            tool_choice="required"  # This forces the LLM to use tools
        )
        
        try:
            response = llm_with_tools.invoke(messages)
            
            # Log the LLM response
            sql_logger.info(f"LLM RESPONSE: {response.content}")
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for i, tool_call in enumerate(response.tool_calls):
                    sql_logger.info(f"TOOL CALL {i+1}: {tool_call}")
            else:
                sql_logger.warning("NO TOOL CALLS DETECTED - This should not happen with tool_choice='required'")
            
            return {"messages": [response]}
            
        except Exception as e:
            sql_logger.error(f"ERROR in _generate_query: {e}")
            # Fallback: Generate SQL directly
            return self._fallback_sql_generation(state)
    
    def _fallback_sql_generation(self, state: MessagesState):
        """Fallback method to generate SQL directly when tool usage fails"""
        user_question = state["messages"][-1].content if state["messages"] else "Unknown"
        sql_logger.info(f"FALLBACK: Generating SQL directly for: {user_question}")
        
        # Generate SQL using LLM without tools
        sql_prompt = f"""
        Generate a SQL query for NYC 311 complaints data to answer this question: {user_question}
        
        Available columns: unique_key, created_date, closed_date, complaint_type, 
        descriptor, status, incident_zip, borough, latitude, longitude, agency
        
        Return ONLY the SQL query, no explanations.
        """
        
        try:
            sql_response = self.llm.invoke([HumanMessage(content=sql_prompt)])
            sql_query = sql_response.content.strip()
            
            # Clean up the SQL query
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            sql_logger.info(f"FALLBACK SQL GENERATED: {sql_query}")
            
            # Execute the SQL query directly
            try:
                result = self.db.run(sql_query)
                sql_logger.info(f"FALLBACK QUERY SUCCESS: {str(result)[:200]}...")
                
                # Create a mock tool call response
                mock_response = AIMessage(
                    content=f"Executed SQL query: {sql_query}",
                    tool_calls=[{
                        "name": "sql_db_query",
                        "args": {"query": sql_query},
                        "id": "fallback_call"
                    }]
                )
                
                return {"messages": [mock_response]}
                
            except Exception as sql_error:
                sql_logger.error(f"FALLBACK SQL EXECUTION FAILED: {sql_error}")
                # Use a simple fallback query
                fallback_query = "SELECT complaint_type, COUNT(*) as count FROM complaints GROUP BY complaint_type ORDER BY count DESC LIMIT 10"
                result = self.db.run(fallback_query)
                
                mock_response = AIMessage(
                    content=f"Used fallback query: {fallback_query}",
                    tool_calls=[{
                        "name": "sql_db_query", 
                        "args": {"query": fallback_query},
                        "id": "fallback_simple"
                    }]
                )
                
                return {"messages": [mock_response]}
                
        except Exception as e:
            sql_logger.error(f"FALLBACK GENERATION FAILED: {e}")
            # Ultimate fallback
            fallback_query = "SELECT complaint_type, COUNT(*) as count FROM complaints GROUP BY complaint_type ORDER BY count DESC LIMIT 10"
            result = self.db.run(fallback_query)
            
            mock_response = AIMessage(
                content="Used ultimate fallback query",
                tool_calls=[{
                    "name": "sql_db_query",
                    "args": {"query": fallback_query}, 
                    "id": "ultimate_fallback"
                }]
            )
            
            return {"messages": [mock_response]}
    
    def _should_continue(self, state: MessagesState) -> Literal["check_query", "format_response"]:
        """Decide whether to check query or format response"""
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "check_query"
        return "format_response"
    
    def _check_query(self, state: MessagesState):
        """Check query, add safety features, and execute SQL returning structured JSON"""
        import sqlite3
        import json
        
        # Extract query from tool call
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            query = last_message.tool_calls[0]["args"]["query"]
            
            # Log the original query
            sql_logger.info(f"ORIGINAL QUERY: {query}")
            
            # Enhanced query safety and validation
            query = self._validate_and_sanitize_query(query)
            sql_logger.info(f"VALIDATED QUERY: {query}")
            
            # Execute query and return structured JSON
            try:
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query(query, conn)
                conn.close()
                
                # Convert to JSON for visualization
                json_payload = df.to_dict(orient="records")
                
                # Truncate if too large (safety measure)
                json_str = json.dumps(json_payload)
                if len(json_str) > 15000:  # Prevent huge responses
                    json_str = json_str[:15000]
                    sql_logger.warning(f"JSON response truncated to 15000 chars")
                
                sql_logger.info(f"QUERY SUCCESS - Rows returned: {len(df)}")
                sql_logger.info(f"QUERY RESULT JSON: {json_str[:500]}...")
                
                # Return structured data for processing
                response = f"SQL_JSON: {json_str}"
                
            except Exception as e:
                sql_logger.error(f"QUERY FAILED: {str(e)}")
                
                # Enhanced fallback with structured data
                fallback_query = "SELECT complaint_type, COUNT(*) as count FROM complaints GROUP BY complaint_type ORDER BY count DESC LIMIT 10"
                sql_logger.info(f"FALLBACK QUERY: {fallback_query}")
                
                try:
                    conn = sqlite3.connect(self.db_path)
                    df = pd.read_sql_query(fallback_query, conn)
                    conn.close()
                    
                    json_payload = df.to_dict(orient="records")
                    json_str = json.dumps(json_payload)
                    
                    sql_logger.info(f"FALLBACK SUCCESS: {len(df)} rows")
                    response = f"SQL_JSON: {json_str}"
                    
                except Exception as fallback_error:
                    sql_logger.error(f"FALLBACK ALSO FAILED: {fallback_error}")
                    response = f"❌ **Both queries failed:**\n\n**Original Error:** {str(e)}\n**Fallback Error:** {str(fallback_error)}\n\nPlease check your database connection."
            
            return {"messages": [AIMessage(content=response)]}
        
        # If no tool calls, something went wrong
        sql_logger.error("NO TOOL CALLS FOUND IN _check_query - This should not happen")
        return {"messages": [AIMessage(content="❌ Error: No SQL query found to execute")]}
    
    def _validate_and_sanitize_query(self, query: str) -> str:
        """Enhanced query validation and safety features"""
        # Clean up the query
        q = query.strip().rstrip(';')
        
        # Must start with SELECT
        if not q.upper().startswith("SELECT"):
            sql_logger.warning("Non-SELECT query detected, using fallback")
            return "SELECT complaint_type, COUNT(*) as count FROM complaints GROUP BY complaint_type ORDER BY count DESC LIMIT 10"
        
        # Strip multiple statements (security)
        if ';' in q:
            q = q.split(';')[0].strip()
            sql_logger.warning("Multiple statements detected, using first statement only")
        
        # Auto-add LIMIT if missing (performance)
        q_upper = q.upper()
        if " LIMIT " not in q_upper:
            q += " LIMIT 50"
            sql_logger.info("Added LIMIT 50 for performance")
        
        # Validate table access (security)
        if "complaints" not in q_upper:
            sql_logger.warning("Query doesn't reference complaints table, using fallback")
            return "SELECT complaint_type, COUNT(*) as count FROM complaints GROUP BY complaint_type ORDER BY count DESC LIMIT 10"
        
        return q
    
    def _execute_query(self, state: MessagesState):
        """Execute the SQL query"""
        # This will be handled by the tool execution
        return state
    
    def _format_response(self, state: MessagesState):
        """Format the final response into natural language with embedded JSON data"""
        import json
        
        last_message = state["messages"][-1]
        
        # Check if this contains structured JSON results
        if last_message.content.startswith("SQL_JSON:"):
            sql_logger.info("Formatting structured JSON results into natural language")
            
            # Extract the JSON data
            json_str = last_message.content.replace("SQL_JSON: ", "", 1)
            
            try:
                # Parse JSON data
                records = json.loads(json_str)
                sql_logger.info(f"Parsed {len(records)} records for formatting")
                
                # Get the original user question from the conversation history
                user_question = "What are the top complaint types?"  # Default fallback
                for msg in state["messages"]:
                    if hasattr(msg, 'content') and isinstance(msg, HumanMessage):
                        user_question = msg.content
                        break
                
                # Use LLM to convert JSON results to natural language
                format_prompt = f"""
                Convert this NYC 311 data analysis result into a clear, natural language response that answers the user's question.
                
                User's Question: {user_question}
                Data Records: {json.dumps(records[:10])}  # Show first 10 for context
                Total Records: {len(records)}
                
                Requirements:
                1. Answer the user's question directly and clearly
                2. Present the data insights in a conversational way
                3. Highlight key findings and patterns
                4. Use bullet points or numbered lists for clarity
                5. Include specific numbers and statistics
                6. Do NOT mention JSON, SQL queries, or technical details
                7. Focus on actionable insights about NYC 311 complaints
                
                Example format:
                "Based on the analysis of {len(records)} records, [direct answer]. Here are the key findings:
                • [Key insight with specific numbers]
                • [Key insight with specific numbers]
                • [Key insight with specific numbers]"
                """
                
                try:
                    formatted_response = self.llm.invoke([HumanMessage(content=format_prompt)])
                    
                    # Create combined response with natural language AND structured data
                    combined_content = f"{formatted_response.content}\n\n[[DATA_JSON]]{json_str}[[/DATA_JSON]]"
                    
                    sql_logger.info("JSON results converted to natural language with embedded data")
                    return {"messages": [AIMessage(content=combined_content)]}
                    
                except Exception as e:
                    sql_logger.error(f"Error formatting with LLM: {e}")
                    # Fallback: create simple natural language response
                    simple_response = self._create_simple_response(records, user_question)
                    combined_content = f"{simple_response}\n\n[[DATA_JSON]]{json_str}[[/DATA_JSON]]"
                    return {"messages": [AIMessage(content=combined_content)]}
                    
            except json.JSONDecodeError as e:
                sql_logger.error(f"Error parsing JSON data: {e}")
                return {"messages": [AIMessage(content="❌ Error parsing data results. Please try your question again.")]}
        
        # For error messages or other responses, return as-is
        elif last_message.content.startswith("❌"):
            sql_logger.info("Returning error message as-is")
            return {"messages": [last_message]}
        
        # For other types of responses, format normally
        else:
            format_prompt = f"""
            Format this response about NYC 311 data into clear, natural language:
            
            {last_message.content}
            
            Make it conversational, helpful, and easy to understand.
            """
            
            try:
                formatted_response = self.llm.invoke([HumanMessage(content=format_prompt)])
                sql_logger.info("Response formatted by LLM")
                return {"messages": [formatted_response]}
            except Exception as e:
                sql_logger.error(f"Error in _format_response: {e}")
                return {"messages": [last_message]}
    
    def _create_simple_response(self, records: list, user_question: str) -> str:
        """Create a simple natural language response from records"""
        if not records:
            return "No data was found for your question."
        
        # Detect common patterns and create appropriate responses
        if len(records) == 1 and len(records[0]) == 1:
            # Single metric
            value = list(records[0].values())[0]
            return f"The result is: {value:,}" if isinstance(value, (int, float)) else f"The result is: {value}"
        
        elif len(records) <= 10 and 'count' in str(records[0]).lower():
            # Top N results
            response = f"Here are the results:\n"
            for i, record in enumerate(records, 1):
                if isinstance(record, dict) and len(record) == 2:
                    key, value = list(record.items())
                    response += f"{i}. {record[key]}: {record[value]:,}\n"
            return response
        
        else:
            # Generic response
            return f"Found {len(records)} results for your question. The data includes various insights about NYC 311 complaints."
    
    def run(self, question: str):
        """Run the agent on a question"""
        initial_state = {
            "messages": [HumanMessage(content=question)]
        }
        
        result = self.graph.invoke(initial_state)
        return result["messages"][-1].content
