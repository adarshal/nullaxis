from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
import logging
import json
import re

logger = logging.getLogger(__name__)

class AnalysisStep(TypedDict):
    """Represents a single analysis step"""
    step_id: str
    description: str
    agent_type: str  # 'sql', 'prebuilt', 'custom', 'transform'
    query: str
    depends_on: List[str]  # List of step_ids this step depends on
    output_type: str  # 'data', 'metric', 'aggregation'

class AnalysisPlan(TypedDict):
    """Complete analysis plan"""
    question: str
    complexity: str  # 'simple', 'moderate', 'complex'
    requires_multi_step: bool
    steps: List[AnalysisStep]
    final_visualization: str  # 'bar', 'pie', 'line', 'table', 'metric'

class QueryPlanner:
    """
    Query Planner Agent using deepseek-reasoner for complex multi-step analysis planning.
    
    This agent analyzes user questions and determines:
    1. Whether the question requires single or multi-step analysis
    2. What specific queries/operations are needed
    3. How to coordinate between different agents
    4. What visualization would be most appropriate
    """
    
    def __init__(self, deepseek_reasoner_client):
        self.reasoner = deepseek_reasoner_client
        
    def analyze_question(self, question: str) -> AnalysisPlan:
        """
        Analyze a user question and create a detailed execution plan.
        
        Uses deepseek-reasoner model for complex reasoning about multi-step analysis.
        """
        planning_prompt = f"""
        You are an expert data analyst planning how to answer questions about NYC 311 complaints data.
        
        Analyze this question and create a detailed execution plan:
        "{question}"
        
        DATABASE SCHEMA:
        - complaint_type: Type of complaint (e.g., 'Blocked Driveway', 'Noise - Street/Sidewalk')
        - borough: NYC borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island)  
        - incident_zip: ZIP code where complaint occurred
        - created_date: When complaint was created
        - closed_date: When complaint was resolved
        - status: Current status of complaint
        - agency: Responsible agency
        - latitude, longitude: Geographic coordinates
        - closed_within_3_days: Boolean - was complaint resolved within 3 days
        - is_geocoded: Boolean - does complaint have valid coordinates
        
        ANALYSIS APPROACH:
        1. Determine if this is a SIMPLE (single query) or COMPLEX (multi-step) question
        2. Break complex questions into logical steps
        3. Choose appropriate agent type for each step:
           - sql: Direct SQL queries for basic aggregations
           - prebuilt: Pre-built analytics functions for common patterns
           - custom: Custom Python analysis for complex statistical operations
           - transform: Data transformation and formatting steps
        4. Plan dependencies between steps
        5. Select the best visualization for the final result
        
        EXAMPLES OF MULTI-STEP QUESTIONS:
        - "For the top 5 complaint types, what percent were closed within 3 days?"
          Step 1: Get top 5 complaint types by count
          Step 2: For those specific types, calculate closure percentage
          
        - "Which Manhattan ZIP codes have the fastest resolution times?"
          Step 1: Filter to Manhattan ZIP codes
          Step 2: Calculate average resolution time by ZIP code
          Step 3: Rank by fastest resolution
          
        - "Make pie chart of top 10 complaint types. Also make line graph of per month complaints count."
          Step 1: Get top 10 complaint types by count (pie chart)
          Step 2: Get monthly complaint counts (line chart)
          Note: Use "multi" as final_visualization for multiple charts
        
        RETURN VALID JSON with this structure:
        {{
            "question": "{question}",
            "complexity": "simple" | "moderate" | "complex",
            "requires_multi_step": boolean,
            "reasoning": "Explanation of why this approach was chosen",
            "steps": [
                {{
                    "step_id": "step_1",
                    "description": "Human readable description",
                    "agent_type": "sql" | "prebuilt" | "custom" | "transform",
                    "query": "SQL query or function call",
                    "depends_on": ["step_id"],
                    "output_type": "data" | "metric" | "aggregation"
                }}
            ],
            "final_visualization": "bar" | "pie" | "line" | "table" | "metric" | "multi"
        }}
        
        Think step by step and create the most efficient plan.
        """
        
        try:
            response = self.reasoner.invoke([HumanMessage(content=planning_prompt)])
            logger.info(f"Query planner reasoning: {response.content[:200]}...")
            
            # Extract JSON from response
            plan_data = self._extract_json_from_response(response.content)
            
            if plan_data:
                # Validate and normalize the plan
                validated_plan = self._validate_plan(plan_data)
                logger.info(f"Created analysis plan with {len(validated_plan['steps'])} steps")
                return validated_plan
            else:
                # Fallback to simple analysis
                logger.warning("Could not parse complex plan, falling back to simple analysis")
                return self._create_simple_fallback_plan(question)
                
        except Exception as e:
            logger.error(f"Error in query planning: {e}")
            return self._create_simple_fallback_plan(question)
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from the reasoner's response"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in reasoner response")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return None
    
    def _validate_plan(self, plan_data: Dict[str, Any]) -> AnalysisPlan:
        """Validate and normalize the analysis plan"""
        # Ensure required fields exist with defaults
        validated = {
            "question": plan_data.get("question", "Unknown question"),
            "complexity": plan_data.get("complexity", "simple"),
            "requires_multi_step": plan_data.get("requires_multi_step", False),
            "steps": [],
            "final_visualization": plan_data.get("final_visualization", "bar")
        }
        
        # Validate steps
        steps = plan_data.get("steps", [])
        for i, step in enumerate(steps):
            validated_step = {
                "step_id": step.get("step_id", f"step_{i+1}"),
                "description": step.get("description", f"Analysis step {i+1}"),
                "agent_type": step.get("agent_type", "sql"),
                "query": step.get("query", "SELECT complaint_type, COUNT(*) FROM complaints GROUP BY complaint_type LIMIT 10"),
                "depends_on": step.get("depends_on", []),
                "output_type": step.get("output_type", "data")
            }
            validated["steps"].append(validated_step)
        
        return validated
    
    def _create_simple_fallback_plan(self, question: str) -> AnalysisPlan:
        """Create a simple single-step plan using LLM-based intent analysis instead of keyword matching"""
        
        # Use LLM to analyze question intent and select appropriate agent
        intent_prompt = f"""
        You are an expert data analyst. Analyze this question about NYC 311 complaints data and determine the best analysis approach.
        
        Question: "{question}"
        
        Available data columns:
        - complaint_type, descriptor, status, agency
        - created_date, closed_date, incident_zip, borough
        - latitude, longitude, closed_within_3_days, is_geocoded
        
        Available analysis approaches:
        1. SQL Agent: For direct database queries and basic aggregations
        2. Prebuilt Analytics: For common patterns like top-k, percentages, time series, geospatial analysis
        3. Custom Code: For complex statistical analysis, correlations, regressions, custom calculations
        4. Data Transform: For data manipulation and formatting
        
        Prebuilt functions available:
        - get_top_k(column, k): Get top K values by count
        - percent_closed_within_days(days, group_col): Calculate closure percentages
        - complaints_by_zip(k): Analyze by ZIP code
        - complaints_by_borough(): Analyze by borough
        - time_series_trend(period): Generate time trends
        - avg_closure_time(group_col): Calculate average closure times
        - geo_validity(): Check geocoding validity
        
        Determine:
        1. What type of analysis is needed (aggregation, statistical, time-series, geospatial, custom)
        2. Which agent would be most appropriate
        3. What specific query or function call would work best
        4. What visualization would be most suitable
        
        Return JSON:
        {{
            "agent_type": "sql|prebuilt|custom|transform",
            "query": "Specific SQL query or function call",
            "description": "Human readable description of what this analysis will do",
            "visualization": "bar|pie|line|table|metric",
            "reasoning": "Why this approach was chosen"
        }}
        """
        
        try:
            response = self.reasoner.invoke([HumanMessage(content=intent_prompt)])
            logger.info(f"Intent analysis: {response.content[:200]}...")
            
            # Extract JSON from response
            intent_data = self._extract_json_from_response(response.content)
            
            if intent_data:
                return {
                    "question": question,
                    "complexity": "simple",
                    "requires_multi_step": False,
                    "steps": [{
                        "step_id": "step_1",
                        "description": intent_data.get("description", f"Analyze: {question}"),
                        "agent_type": intent_data.get("agent_type", "sql"),
                        "query": intent_data.get("query", "SELECT complaint_type, COUNT(*) as count FROM complaints GROUP BY complaint_type ORDER BY count DESC LIMIT 10"),
                        "depends_on": [],
                        "output_type": "data"
                    }],
                    "final_visualization": intent_data.get("visualization", "bar")
                }
            else:
                # Ultimate fallback if LLM fails
                logger.warning("LLM intent analysis failed, using basic fallback")
                return self._create_basic_fallback_plan(question)
                
        except Exception as e:
            logger.error(f"Error in LLM intent analysis: {e}")
            return self._create_basic_fallback_plan(question)
    
    def _create_basic_fallback_plan(self, question: str) -> AnalysisPlan:
        """Ultimate fallback plan with minimal hardcoded logic"""
        return {
            "question": question,
            "complexity": "simple",
            "requires_multi_step": False,
            "steps": [{
                "step_id": "step_1",
                "description": f"Analyze: {question}",
                "agent_type": "sql",
                "query": "SELECT complaint_type, COUNT(*) as count FROM complaints GROUP BY complaint_type ORDER BY count DESC LIMIT 10",
                "depends_on": [],
                "output_type": "data"
            }],
            "final_visualization": "bar"
        }
    
    def detect_question_patterns(self, question: str) -> Dict[str, Any]:
        """Detect question patterns using LLM-based analysis instead of regex patterns"""
        pattern_prompt = f"""
        Analyze this question about NYC 311 complaints data to detect analysis patterns:
        
        Question: "{question}"
        
        Determine if this question requires:
        1. Multi-step analysis (complex operations that need multiple queries/steps)
        2. Single-step analysis (can be answered with one query/operation)
        3. What type of analysis pattern it follows
        
        Return JSON:
        {{
            "requires_multi_step": true/false,
            "complexity": "simple|moderate|complex",
            "analysis_type": "aggregation|comparison|trend|statistical|geospatial|custom",
            "reasoning": "Why this classification was chosen"
        }}
        """
        
        try:
            response = self.reasoner.invoke([HumanMessage(content=pattern_prompt)])
            pattern_data = self._extract_json_from_response(response.content)
            
            if pattern_data:
                return {
                    "detected_patterns": [{"name": pattern_data.get("analysis_type", "unknown"), "requires_multi_step": pattern_data.get("requires_multi_step", False)}],
                    "requires_multi_step": pattern_data.get("requires_multi_step", False),
                    "complexity": pattern_data.get("complexity", "simple")
                }
            else:
                # Fallback to simple analysis
                return {
                    "detected_patterns": [],
                    "requires_multi_step": False,
                    "complexity": "simple"
                }
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
        return {
                "detected_patterns": [],
                "requires_multi_step": False,
                "complexity": "simple"
        }
