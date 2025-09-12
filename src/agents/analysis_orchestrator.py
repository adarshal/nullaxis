from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
import logging
import json
import pandas as pd
from .query_planner import QueryPlanner, AnalysisPlan, AnalysisStep
from .sql_agent import NYC311SQLAgent
from ..executors.prebuilt_functions import PrebuiltAnalytics
from ..executors.code_writer import CodeWriter

logger = logging.getLogger(__name__)

class StepResult(Dict):
    """Result from executing an analysis step"""
    step_id: str
    success: bool
    data: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    metadata: Dict[str, Any]

class AnalysisOrchestrator:
    """
    Analysis Orchestrator coordinates multi-step analysis workflows.
    
    This agent:
    1. Takes plans from the Query Planner
    2. Executes steps in the correct order
    3. Manages dependencies between steps
    4. Combines results for final visualization
    5. Handles errors and fallbacks
    """
    
    def __init__(self, deepseek_client, sql_agent: NYC311SQLAgent, db_path: str):
        self.deepseek_client = deepseek_client
        self.sql_agent = sql_agent
        self.db_path = db_path
        self.query_planner = QueryPlanner(deepseek_client.reasoner)
        
        # Initialize advanced analysis components
        self.prebuilt_analytics = PrebuiltAnalytics(db_path)
        self.code_writer = CodeWriter(db_path)
        
        # Results cache for multi-step workflows
        self.step_results: Dict[str, StepResult] = {}
    
    def execute_analysis(self, question: str) -> Dict[str, Any]:
        """
        Execute complete analysis workflow for a user question.
        
        Returns:
        {
            "natural_language_response": "Human readable answer",
            "data": [...],  # Final data for visualization
            "chart_type": "bar|pie|line|table|metric",
            "execution_steps": [...],  # Details of what was executed
            "success": True/False
        }
        """
        logger.info(f"Starting analysis orchestration for: {question}")
        
        try:
            # Step 1: Plan the analysis
            plan = self.query_planner.analyze_question(question)
            logger.info(f"Created {plan['complexity']} plan with {len(plan['steps'])} steps")
            
            # Step 2: Execute the plan
            if plan["requires_multi_step"]:
                return self._execute_multi_step_plan(plan)
            else:
                return self._execute_simple_plan(plan)
                
        except Exception as e:
            logger.error(f"Analysis orchestration failed: {e}")
            return {
                "natural_language_response": f"I encountered an error while analyzing your question: {str(e)}",
                "data": [],
                "chart_type": "error",
                "execution_steps": [],
                "success": False
            }
    
    def _execute_simple_plan(self, plan: AnalysisPlan) -> Dict[str, Any]:
        """Execute a simple single-step plan"""
        logger.info("Executing simple analysis plan")
        
        step = plan["steps"][0]
        result = self._execute_step(step, {})
        
        if result["success"]:
            # Generate natural language response
            nl_response = self._generate_natural_language_response(
                plan["question"], 
                result["data"], 
                plan["complexity"]
            )
            
            return {
                "natural_language_response": nl_response,
                "data": result["data"],
                "chart_type": plan["final_visualization"],
                "execution_steps": [result],
                "success": True
            }
        else:
            return {
                "natural_language_response": f"I couldn't analyze your question: {result.get('error', 'Unknown error')}",
                "data": [],
                "chart_type": "error",
                "execution_steps": [result],
                "success": False
            }
    
    def _execute_multi_step_plan(self, plan: AnalysisPlan) -> Dict[str, Any]:
        """Execute a complex multi-step plan with dependency management"""
        logger.info(f"Executing multi-step analysis plan with {len(plan['steps'])} steps")
        
        self.step_results = {}  # Reset results cache
        executed_steps = []
        final_data = []
        
        # Execute steps in dependency order
        for step in plan["steps"]:
            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(step, self.step_results):
                logger.error(f"Dependencies not satisfied for step {step['step_id']}")
                continue
            
            # Execute step with context from previous steps
            context = self._build_step_context(step, self.step_results)
            result = self._execute_step(step, context)
            
            # Store result
            self.step_results[step["step_id"]] = result
            executed_steps.append(result)
            
            # If step failed, try to continue or abort
            if not result["success"]:
                logger.warning(f"Step {step['step_id']} failed: {result.get('error')}")
                # For now, continue with other steps
        
        # Combine results for final response
        if executed_steps:
            # Use the last successful step's data as primary result
            final_result = None
            for step_result in reversed(executed_steps):
                if step_result["success"] and step_result["data"]:
                    final_result = step_result
                    break
            
            if final_result:
                # Generate comprehensive natural language response
                nl_response = self._generate_multi_step_response(
                    plan["question"], 
                    executed_steps, 
                    plan["complexity"]
                )
                
                return {
                    "natural_language_response": nl_response,
                    "data": final_result["data"],
                    "chart_type": plan["final_visualization"],
                    "execution_steps": executed_steps,
                    "success": True
                }
        
        # Fallback if all steps failed
        return {
            "natural_language_response": "I was unable to complete the multi-step analysis. Please try a simpler question.",
            "data": [],
            "chart_type": "error",
            "execution_steps": executed_steps,
            "success": False
        }
    
    def _execute_step(self, step: AnalysisStep, context: Dict[str, Any]) -> StepResult:
        """Execute a single analysis step"""
        logger.info(f"Executing step {step['step_id']}: {step['description']}")
        
        try:
            if step["agent_type"] == "sql":
                logger.info("🔧 Using SQL Agent for execution")
                return self._execute_sql_step(step, context)
            elif step["agent_type"] == "prebuilt":
                logger.info("Using PrebuiltAnalytics executor")
                return self._execute_prebuilt_step(step, context)
            elif step["agent_type"] == "custom":
                logger.info("CODEWRITER: Using CodeWriter executor for custom Python analysis")
                return self._execute_custom_step(step, context)
            elif step["agent_type"] == "transform":
                logger.info("Using data transformation")
                return self._execute_transform_step(step, context)
            
            else:
                return {
                    "step_id": step["step_id"],
                    "success": False,
                    "data": None,
                    "error": f"Unknown agent type: {step['agent_type']}",
                    "metadata": {"step": step}
                }
                
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {
                "step_id": step["step_id"],
                "success": False,
                "data": None,
                "error": str(e),
                "metadata": {"step": step}
            }
    
    def _execute_sql_step(self, step: AnalysisStep, context: Dict[str, Any]) -> StepResult:
        """Execute a SQL-based analysis step"""
        # Modify query based on context if needed
        query = step["query"]
        
        # If this step depends on previous steps, modify the query
        if context and step["depends_on"]:
            query = self._adapt_query_with_context(query, context)
        else:
            # Still ensure SQLite compatibility for standalone queries
            query = self._make_sqlite_compatible(query)
        
        # Use the existing SQL agent to execute
        try:
            # Create a temporary question for the SQL agent
            temp_question = step["description"]
            
            # Temporarily override the SQL agent to use our specific query
            original_run = self.sql_agent.run
            
            def custom_run(question):
                # Execute our specific query instead
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query(query, conn)
                conn.close()
                
                json_data = df.to_dict('records')
                
                # Format as the SQL agent would
                nl_response = f"Query executed successfully. Found {len(json_data)} results."
                return f"{nl_response}\n\n[[DATA_JSON]]{json.dumps(json_data)}[[/DATA_JSON]]"
            
            # Replace temporarily
            self.sql_agent.run = custom_run
            
            try:
                response = self.sql_agent.run(temp_question)
                
                # Extract data
                import re
                json_match = re.search(r"\[\[DATA_JSON\]\](.+?)\[\[/DATA_JSON\]\]", response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1).strip())
                    
                    return {
                        "step_id": step["step_id"],
                        "success": True,
                        "data": data,
                        "error": None,
                        "metadata": {"query": query, "row_count": len(data)}
                    }
                else:
                    return {
                        "step_id": step["step_id"],
                        "success": False,
                        "data": None,
                        "error": "Could not extract data from SQL result",
                        "metadata": {"query": query}
                    }
                    
            finally:
                # Restore original method
                self.sql_agent.run = original_run
                
        except Exception as e:
            return {
                "step_id": step["step_id"],
                "success": False,
                "data": None,
                "error": f"SQL execution failed: {str(e)}",
                "metadata": {"query": query}
            }
    
    def _execute_prebuilt_step(self, step: AnalysisStep, context: Dict[str, Any]) -> StepResult:
        """Execute a prebuilt analytics function"""
        logger.info(f"Executing prebuilt analytics step: {step['query']}")
        
        try:
            # Parse function call from query
            function_call = step["query"]
            
            # Map common queries to prebuilt functions
            function_mappings = {
                "get_top_k": self.prebuilt_analytics.get_top_k,
                "percent_closed_within_days": self.prebuilt_analytics.percent_closed_within_days,
                "complaints_by_zip": self.prebuilt_analytics.complaints_by_zip,
                "complaints_by_borough": self.prebuilt_analytics.complaints_by_borough,
                "geo_validity": self.prebuilt_analytics.geo_validity,
                "time_series_trend": self.prebuilt_analytics.time_series_trend,
                "avg_closure_time": self.prebuilt_analytics.avg_closure_time,
                "complaints_by_agency": self.prebuilt_analytics.complaints_by_agency,
                "open_vs_closed": self.prebuilt_analytics.open_vs_closed
            }
            
            # Try to match the function call
            for func_name, func in function_mappings.items():
                if func_name in function_call.lower():
                    # Extract parameters (simple implementation)
                    if "top_k" in function_call.lower() and "complaint_type" in function_call.lower():
                        result = self.prebuilt_analytics.get_top_k("complaint_type", 10)
                    elif "closed_within" in function_call.lower():
                        result = self.prebuilt_analytics.percent_closed_within_days(3, "complaint_type")
                    elif "by_zip" in function_call.lower():
                        result = self.prebuilt_analytics.complaints_by_zip(10)
                    elif "by_borough" in function_call.lower():
                        result = self.prebuilt_analytics.complaints_by_borough()
                    elif "time_series" in function_call.lower() or "trend" in function_call.lower():
                        result = self.prebuilt_analytics.time_series_trend("month")
                    else:
                        # Default to the first matched function
                        result = func()
                    
                    if "error" not in result:
                        return {
                            "step_id": step["step_id"],
                            "success": True,
                            "data": result["data"],
                            "error": None,
                            "metadata": {
                                "step": step,
                                "function": func_name,
                                "row_count": len(result["data"]) if result["data"] else 0,
                                "chart_type": result.get("chart_type", "bar")
                            }
                        }
                    else:
                        return {
                            "step_id": step["step_id"],
                            "success": False,
                            "data": None,
                            "error": result["error"],
                            "metadata": {"step": step, "function": func_name}
                        }
            
            # If no function matched, fall back to SQL execution
            logger.warning(f"No prebuilt function matched for: {function_call}, falling back to SQL")
            return self._execute_sql_step(step, context)
            
        except Exception as e:
            logger.error(f"Prebuilt analytics step failed: {e}")
            return {
                "step_id": step["step_id"],
                "success": False,
                "data": None,
                "error": f"Prebuilt analytics execution failed: {str(e)}",
                "metadata": {"step": step}
            }
    
    def _execute_custom_step(self, step: AnalysisStep, context: Dict[str, Any]) -> StepResult:
        """Execute custom Python analysis using CodeWriter"""
        logger.info(f"Executing custom Python analysis: {step['description']}")
        
        try:
            # Generate Python code based on the step description and context
            code_prompt = self._generate_custom_code_from_step(step, context)
            
            # Execute custom code
            result = self.code_writer.execute_custom_code(code_prompt, step["query"])
            
            if "error" not in result:
                # Convert result to standard format
                data = []
                if result.get("result") is not None:
                    if isinstance(result["result"], pd.DataFrame):
                        data = result["result"].to_dict('records')
                    elif isinstance(result["result"], list):
                        data = result["result"]
                    elif isinstance(result["result"], dict):
                        data = [result["result"]]
                    else:
                        # Single value result
                        data = [{"result": result["result"]}]
                
                return {
                    "step_id": step["step_id"],
                    "success": True,
                    "data": data,
                    "error": None,
                    "metadata": {
                        "step": step,
                        "code_executed": code_prompt,
                        "output": result.get("output", ""),
                        "row_count": len(data) if data else 0,
                        "chart_type": result.get("chart_type", "table")
                    }
                }
            else:
                return {
                    "step_id": step["step_id"],
                    "success": False,
                    "data": None,
                    "error": result["error"],
                    "metadata": {
                        "step": step,
                        "code_attempted": code_prompt,
                        "traceback": result.get("traceback", "")
                    }
                }
                
        except Exception as e:
            logger.error(f"Custom analysis step failed: {e}")
            return {
                "step_id": step["step_id"],
                "success": False,
                "data": None,
                "error": f"Custom analysis execution failed: {str(e)}",
                "metadata": {"step": step}
            }
    
    def _generate_custom_code_from_step(self, step: AnalysisStep, context: Dict[str, Any]) -> str:
        """Generate Python code for custom analysis based on step description and context"""
        description = step["description"]
        
        # Start with basic template
        code = """
# Custom analysis for: {description}

""".format(description=description)
        
        # Add context data if available
        if context:
            code += "# Context from previous steps:\n"
            for step_id, data in context.items():
                if data:
                    code += f"# {step_id}: {len(data)} records\n"
        
        # Generate specific code based on description patterns
        description_lower = description.lower()
        
        if "seasonal" in description_lower and "trend" in description_lower:
            code += """
# Analyze seasonal trends
df['created_date'] = pd.to_datetime(df['created_date'])
df['month'] = df['created_date'].dt.month
df['year'] = df['created_date'].dt.year

# Group by month and complaint type for seasonal analysis
seasonal_data = df.groupby(['month', 'complaint_type']).size().reset_index(name='count')

# Focus on top complaint types if specified in context
if len(seasonal_data) > 100:  # If too much data, focus on top types
    top_types = df.groupby('complaint_type').size().nlargest(3).index.tolist()
    seasonal_data = seasonal_data[seasonal_data['complaint_type'].isin(top_types)]

result = seasonal_data.to_dict('records')
"""
        
        elif "percent" in description_lower and "closed" in description_lower:
            code += """
# Calculate closure percentages
df['closure_days'] = (pd.to_datetime(df['closed_date']) - pd.to_datetime(df['created_date'])).dt.days
df['closed_within_3_days'] = (df['closure_days'] <= 3) & (df['closure_days'] >= 0)

# Group by complaint type
closure_stats = df.groupby('complaint_type').agg({
    'closed_within_3_days': ['count', 'sum'],
    'unique_key': 'count'
}).round(2)

closure_stats.columns = ['total_closed_3days', 'count_closed_3days', 'total_complaints']
closure_stats['percentage_closed_3days'] = (
    closure_stats['count_closed_3days'] / closure_stats['total_complaints'] * 100
).round(2)

result = closure_stats.reset_index().to_dict('records')
"""
        
        elif "fastest" in description_lower or "resolution" in description_lower:
            code += """
# Calculate resolution times
df['resolution_days'] = (pd.to_datetime(df['closed_date']) - pd.to_datetime(df['created_date'])).dt.days
df_closed = df[df['closed_date'].notna() & (df['resolution_days'] >= 0)]

# Group by the relevant dimension (zip, borough, etc.)
group_col = 'incident_zip' if 'zip' in description.lower() else 'borough'
resolution_stats = df_closed.groupby(group_col).agg({
    'resolution_days': ['mean', 'count']
}).round(2)

resolution_stats.columns = ['avg_resolution_days', 'complaint_count']
resolution_stats = resolution_stats[resolution_stats['complaint_count'] >= 10]  # Filter for significance

result = resolution_stats.sort_values('avg_resolution_days').reset_index().to_dict('records')
"""
        
        else:
            # Generic aggregation
            code += """
# Generic data analysis
if 'complaint_type' in df.columns:
    result = df.groupby('complaint_type').size().sort_values(ascending=False).head(10).reset_index(name='count').to_dict('records')
else:
    result = df.head(10).to_dict('records')
"""
        
        return code
    
    def _execute_transform_step(self, step: AnalysisStep, context: Dict[str, Any]) -> StepResult:
        """Execute data transformation step"""
        logger.info(f"Executing data transformation: {step['description']}")
        
        try:
            # Get data from previous steps
            input_data = []
            for dep_id in step.get("depends_on", []):
                if dep_id in context:
                    input_data.extend(context[dep_id])
            
            if not input_data:
                return {
                    "step_id": step["step_id"],
                    "success": False,
                    "data": None,
                    "error": "No input data available for transformation",
                    "metadata": {"step": step}
                }
            
            # Convert to DataFrame for processing
            df = pd.DataFrame(input_data)
            
            # Apply transformations based on step description
            description_lower = step["description"].lower()
            transformed_data = df
            
            if "combine" in description_lower or "merge" in description_lower:
                # Simple combination of results
                transformed_data = df.groupby('complaint_type').agg({
                    'count': 'sum'
                }).reset_index() if 'complaint_type' in df.columns and 'count' in df.columns else df
                
            elif "filter" in description_lower:
                # Filter top N results
                if 'count' in df.columns:
                    transformed_data = df.nlargest(10, 'count')
                else:
                    transformed_data = df.head(10)
                    
            elif "percentage" in description_lower:
                # Add percentage calculations
                if 'count' in df.columns:
                    total = df['count'].sum()
                    transformed_data = df.copy()
                    transformed_data['percentage'] = (df['count'] / total * 100).round(2)
                    
            elif "format" in description_lower:
                # Format for visualization
                if len(df.columns) >= 2:
                    # Ensure standard column names for charting
                    transformed_data = df.copy()
                    if df.columns.tolist() != ['category', 'value']:
                        transformed_data.columns = ['category', 'value'][:len(df.columns)]
            
            result_data = transformed_data.to_dict('records')
            
            return {
                "step_id": step["step_id"],
                "success": True,
                "data": result_data,
                "error": None,
                "metadata": {
                    "step": step,
                    "input_rows": len(input_data),
                    "output_rows": len(result_data),
                    "transformation_type": "data_transformation"
                }
            }
            
        except Exception as e:
            logger.error(f"Data transformation step failed: {e}")
            return {
                "step_id": step["step_id"],
                "success": False,
                "data": None,
                "error": f"Data transformation failed: {str(e)}",
                "metadata": {"step": step}
            }
    
    def _dependencies_satisfied(self, step: AnalysisStep, results: Dict[str, StepResult]) -> bool:
        """Check if step dependencies are satisfied"""
        # If no explicit dependencies, allow execution (will use all previous results)
        if not step.get("depends_on"):
            return True
        
        # Check explicit dependencies
        for dep_id in step["depends_on"]:
            if dep_id not in results or not results[dep_id]["success"]:
                logger.warning(f"Dependency {dep_id} not satisfied for step {step['step_id']}")
                return False
        
        logger.info(f"All dependencies satisfied for step {step['step_id']}")
        return True
    
    def _build_step_context(self, step: AnalysisStep, results: Dict[str, StepResult]) -> Dict[str, Any]:
        """Build context for step execution from previous results"""
        context = {}
        
        # Add successful previous step results to context
        for dep_id in step["depends_on"]:
            if dep_id in results and results[dep_id]["success"]:
                context[dep_id] = results[dep_id]["data"]
                logger.info(f"Added {dep_id} to context with {len(results[dep_id]['data']) if results[dep_id]['data'] else 0} records")
        
        # Also include all previous successful steps for potential use
        # This helps with cases where the step doesn't explicitly list dependencies
        for step_id, result in results.items():
            if result["success"] and step_id not in context:
                context[step_id] = result["data"]
                logger.info(f"Added {step_id} to context as available data with {len(result['data']) if result['data'] else 0} records")
        
        logger.info(f"Built context with {len(context)} data sources: {list(context.keys())}")
        return context
    
    def _adapt_query_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """Adapt SQL query based on context from previous steps"""
        logger.info(f"Adapting query with context: {list(context.keys())}")
        
        adapted_query = query
        
        # Replace step references with actual data
        for step_id, data in context.items():
            if data and isinstance(data, list):
                # Handle different types of context injection
                
                # Case 1: IN clause with specific values
                step_table_ref = f"{step_id}_output"
                if step_table_ref in adapted_query:
                    # Extract values from the context data
                    if data and isinstance(data[0], dict):
                        # Try to find the right column to extract
                        first_record = data[0]
                        
                        # For complaint types, look for complaint_type field
                        if 'complaint_type' in first_record:
                            complaint_types = [f"'{record['complaint_type']}'" for record in data]
                            values_list = "(" + ", ".join(complaint_types) + ")"
                            adapted_query = adapted_query.replace(
                                f"(SELECT complaint_type FROM {step_table_ref})", 
                                values_list
                            )
                            logger.info(f"Replaced {step_table_ref} with {len(complaint_types)} complaint types")
                        
                        # For ZIP codes
                        elif 'incident_zip' in first_record:
                            zip_codes = [f"'{record['incident_zip']}'" for record in data if record['incident_zip']]
                            values_list = "(" + ", ".join(zip_codes) + ")"
                            adapted_query = adapted_query.replace(
                                f"(SELECT incident_zip FROM {step_table_ref})",
                                values_list
                            )
                            logger.info(f"Replaced {step_table_ref} with {len(zip_codes)} ZIP codes")
                        
                        # For boroughs
                        elif 'borough' in first_record:
                            boroughs = [f"'{record['borough']}'" for record in data if record['borough']]
                            values_list = "(" + ", ".join(boroughs) + ")"
                            adapted_query = adapted_query.replace(
                                f"(SELECT borough FROM {step_table_ref})",
                                values_list
                            )
                            logger.info(f"Replaced {step_table_ref} with {len(boroughs)} boroughs")
                        
                        # Generic fallback - use first column
                        else:
                            first_col = list(first_record.keys())[0]
                            values = [f"'{record[first_col]}'" for record in data if record[first_col]]
                            if values:
                                values_list = "(" + ", ".join(values) + ")"
                                adapted_query = adapted_query.replace(
                                    f"(SELECT {first_col} FROM {step_table_ref})",
                                    values_list
                                )
                                logger.info(f"Replaced {step_table_ref} with {len(values)} values from {first_col}")
        
        logger.info(f"Adapted query: {adapted_query}")
        
        # Make sure query is SQLite compatible
        adapted_query = self._make_sqlite_compatible(adapted_query)
        logger.info(f"SQLite-compatible query: {adapted_query}")
        
        return adapted_query
    
    def _make_sqlite_compatible(self, query: str) -> str:
        """Convert SQL query to be SQLite compatible"""
        sqlite_query = query
        
        # Replace PostgreSQL EXTRACT with SQLite julianday functions
        import re
        
        # Pattern: EXTRACT(EPOCH FROM (date1 - date2)) -> (julianday(date1) - julianday(date2)) * 86400
        epoch_pattern = r'EXTRACT\(EPOCH\s+FROM\s+\(([^)]+)\s*-\s*([^)]+)\)\)'
        sqlite_query = re.sub(epoch_pattern, r'((julianday(\1) - julianday(\2)) * 86400)', sqlite_query, flags=re.IGNORECASE)
        
        # Pattern: EXTRACT(EPOCH FROM date_col) -> julianday(date_col) * 86400
        epoch_pattern2 = r'EXTRACT\(EPOCH\s+FROM\s+([^)]+)\)'
        sqlite_query = re.sub(epoch_pattern2, r'(julianday(\1) * 86400)', sqlite_query, flags=re.IGNORECASE)
        
        # Convert hours calculation: / 3600 -> / 3600.0
        sqlite_query = sqlite_query.replace('/ 3600', '/ 3600.0')
        
        # Convert days calculation for resolution time
        resolution_pattern = r'\(([^)]*closed_date[^)]*)\s*-\s*([^)]*created_date[^)]*)\)\s*/\s*3600\.0'
        sqlite_query = re.sub(resolution_pattern, r'((julianday(\1) - julianday(\2)) * 24)', sqlite_query, flags=re.IGNORECASE)
        
        # More specific pattern for our date difference in hours
        if 'resolution_hours' in sqlite_query:
            # Replace the entire EXTRACT pattern specifically for our use case
            sqlite_query = re.sub(
                r'EXTRACT\(EPOCH FROM \(closed_date - created_date\)\) / 3600',
                '((julianday(closed_date) - julianday(created_date)) * 24)',
                sqlite_query,
                flags=re.IGNORECASE
            )
        
        # Handle boolean conversions (TRUE/FALSE to 1/0)
        sqlite_query = sqlite_query.replace('TRUE', '1').replace('FALSE', '0')
        
        # Fix string quoting issues (ensure single quotes)
        sqlite_query = sqlite_query.replace('"', "'")
        
        logger.info(f"Converted to SQLite: {sqlite_query}")
        return sqlite_query
    
    def _generate_natural_language_response(self, question: str, data: List[Dict], complexity: str) -> str:
        """Generate natural language response for simple analysis using LLM"""
        if not data:
            return "No data was found for your question."
        
        # Use LLM to generate a meaningful response
        try:
            data_summary = json.dumps(data[:10], indent=2)  # Limit to top 10 for context
            
            prompt = f"""You are a data analyst. Based on the following data from NYC 311 complaints, provide a clear, insightful response to the user's question.

User Question: "{question}"

Data:
{data_summary}

Please provide:
1. A direct answer to their question
2. Key insights from the data
3. Notable patterns or findings
4. Specific numbers and statistics

Keep it concise but informative. Focus on what the user actually asked for."""

            response = self.deepseek_client.client.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM response generation failed: {e}, falling back to basic response")
            # Fallback to basic response
            if len(data) == 1 and len(data[0]) == 1:
                value = list(data[0].values())[0]
                return f"The result is: {value:,}" if isinstance(value, (int, float)) else f"The result is: {value}"
            elif len(data) <= 10:
                response = f"Based on the analysis, here are the key findings:\n"
                for i, record in enumerate(data[:5], 1):
                    if len(record) == 2:
                        keys = list(record.keys())
                        response += f"• {record[keys[0]]}: {record[keys[1]]:,}\n"
                return response
            return f"Found {len(data)} results for your question about NYC 311 complaints."
    
    def _generate_multi_step_response(self, question: str, steps: List[StepResult], complexity: str) -> str:
        """Generate comprehensive natural language response for multi-step analysis using LLM"""
        successful_steps = [s for s in steps if s["success"]]
        
        if not successful_steps:
            return "I was unable to complete the analysis steps for your question."
        
        # Use LLM to generate a comprehensive response
        try:
            # Prepare step summaries for context
            step_summaries = []
            for i, step in enumerate(successful_steps, 1):
                if step["data"]:
                    step_data = step["data"][:5]  # Limit to top 5 for context
                    step_summaries.append(f"Step {i}: {json.dumps(step_data, indent=2)}")
            
            # Get final results
            final_step = successful_steps[-1]
            final_data = final_step["data"][:10] if final_step["data"] else []
            
            context = "\n\n".join(step_summaries)
            final_data_json = json.dumps(final_data, indent=2)
            
            prompt = f"""You are a data analyst. I completed a multi-step analysis for a user's question about NYC 311 complaints. 

User Question: "{question}"

Analysis Steps Completed:
{context}

Final Results:
{final_data_json}

Please provide a comprehensive response that:
1. Directly answers the user's question
2. Explains the key findings from each step
3. Highlights the most important insights
4. Provides specific numbers and statistics
5. Explains what the data means in practical terms

Make it engaging and informative, focusing on what the user actually asked for."""

            response = self.deepseek_client.client.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM multi-step response generation failed: {e}, falling back to basic response")
            # Fallback to basic response
            response = f"I completed a {complexity} analysis with {len(successful_steps)} steps:\n\n"
            
            for i, step in enumerate(successful_steps, 1):
                if step["data"]:
                    response += f"**Step {i}:** Found {len(step['data'])} results\n"
            
            # Add summary from the final step
            final_step = successful_steps[-1]
            if final_step["data"]:
                response += f"\n**Final Results:** {self._generate_natural_language_response(question, final_step['data'], complexity)}"
            
            return response
