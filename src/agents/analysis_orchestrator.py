from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
import logging
import json
import pandas as pd
from .query_planner import QueryPlanner, AnalysisPlan, AnalysisStep
from .sql_agent import NYC311SQLAgent

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
                return self._execute_sql_step(step, context)
            elif step["agent_type"] == "prebuilt":
                return self._execute_prebuilt_step(step, context)
            elif step["agent_type"] == "custom":
                return self._execute_custom_step(step, context)
            elif step["agent_type"] == "transform":
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
        # TODO: Integrate with PrebuiltAnalytics
        return {
            "step_id": step["step_id"],
            "success": False,
            "data": None,
            "error": "Prebuilt analytics not yet integrated",
            "metadata": {"step": step}
        }
    
    def _execute_custom_step(self, step: AnalysisStep, context: Dict[str, Any]) -> StepResult:
        """Execute custom Python analysis"""
        # TODO: Integrate with CodeWriter
        return {
            "step_id": step["step_id"],
            "success": False,
            "data": None,
            "error": "Custom analysis not yet integrated",
            "metadata": {"step": step}
        }
    
    def _execute_transform_step(self, step: AnalysisStep, context: Dict[str, Any]) -> StepResult:
        """Execute data transformation step"""
        # TODO: Implement data transformation logic
        return {
            "step_id": step["step_id"],
            "success": False,
            "data": None,
            "error": "Data transformation not yet implemented",
            "metadata": {"step": step}
        }
    
    def _dependencies_satisfied(self, step: AnalysisStep, results: Dict[str, StepResult]) -> bool:
        """Check if step dependencies are satisfied"""
        for dep_id in step["depends_on"]:
            if dep_id not in results or not results[dep_id]["success"]:
                return False
        return True
    
    def _build_step_context(self, step: AnalysisStep, results: Dict[str, StepResult]) -> Dict[str, Any]:
        """Build context for step execution from previous results"""
        context = {}
        for dep_id in step["depends_on"]:
            if dep_id in results and results[dep_id]["success"]:
                context[dep_id] = results[dep_id]["data"]
        return context
    
    def _adapt_query_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """Adapt SQL query based on context from previous steps"""
        # This is a simplified implementation
        # In a more sophisticated system, you'd have more complex query modification logic
        return query
    
    def _generate_natural_language_response(self, question: str, data: List[Dict], complexity: str) -> str:
        """Generate natural language response for simple analysis"""
        if not data:
            return "No data was found for your question."
        
        # Basic response generation
        if len(data) == 1 and len(data[0]) == 1:
            # Single metric
            value = list(data[0].values())[0]
            return f"The result is: {value:,}" if isinstance(value, (int, float)) else f"The result is: {value}"
        
        elif len(data) <= 10:
            # Top N results
            response = f"Based on the analysis, here are the key findings:\n"
            for i, record in enumerate(data[:5], 1):  # Show top 5
                if len(record) == 2:
                    keys = list(record.keys())
                    response += f"• {record[keys[0]]}: {record[keys[1]]:,}\n"
            return response
        
        return f"Found {len(data)} results for your question about NYC 311 complaints."
    
    def _generate_multi_step_response(self, question: str, steps: List[StepResult], complexity: str) -> str:
        """Generate comprehensive natural language response for multi-step analysis"""
        successful_steps = [s for s in steps if s["success"]]
        
        if not successful_steps:
            return "I was unable to complete the analysis steps for your question."
        
        response = f"I completed a {complexity} analysis with {len(successful_steps)} steps:\n\n"
        
        for i, step in enumerate(successful_steps, 1):
            if step["data"]:
                response += f"**Step {i}:** Found {len(step['data'])} results\n"
        
        # Add summary from the final step
        final_step = successful_steps[-1]
        if final_step["data"]:
            response += f"\n**Final Results:** {self._generate_natural_language_response(question, final_step['data'], complexity)}"
        
        return response
