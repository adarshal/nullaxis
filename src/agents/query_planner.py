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
            "final_visualization": "bar" | "pie" | "line" | "table" | "metric"
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
        """Create a simple single-step plan as fallback using advanced capabilities"""
        question_lower = question.lower()
        
        # Enhanced intent detection with intelligent agent selection
        chart_type = "bar"  # default
        agent_type = "sql"  # default
        query = "SELECT complaint_type, COUNT(*) as count FROM complaints GROUP BY complaint_type ORDER BY count DESC LIMIT 10"
        description = f"Analyze: {question}"
        
        # Use prebuilt analytics for common patterns
        if any(word in question_lower for word in ['top', 'most', 'highest']) and 'complaint' in question_lower:
            agent_type = "prebuilt"
            query = "get_top_k('complaint_type', 10)"
            description = "Get top complaint types using prebuilt analytics"
            
        elif any(word in question_lower for word in ['percent', 'percentage', 'proportion']):
            chart_type = "pie"
            if 'closed within 3 days' in question_lower:
                agent_type = "prebuilt"
                query = "percent_closed_within_days(3, 'complaint_type')"
                description = "Calculate closure percentages using prebuilt analytics"
            elif 'geocoded' in question_lower or 'latitude' in question_lower or 'longitude' in question_lower:
                agent_type = "prebuilt"
                query = "geo_validity()"
                description = "Check geocoding validity using prebuilt analytics"
                
        elif any(word in question_lower for word in ['zip', 'zipcode', 'zip code']):
            agent_type = "prebuilt"
            query = "complaints_by_zip(10)"
            description = "Analyze complaints by ZIP code using prebuilt analytics"
            
        elif 'borough' in question_lower:
            chart_type = "pie"
            agent_type = "prebuilt"
            query = "complaints_by_borough()"
            description = "Analyze complaints by borough using prebuilt analytics"
            
        elif any(word in question_lower for word in ['trend', 'time', 'seasonal', 'monthly']):
            chart_type = "line"
            agent_type = "prebuilt"
            query = "time_series_trend('month')"
            description = "Generate time series trends using prebuilt analytics"
            
        elif any(word in question_lower for word in ['resolution', 'closure', 'response']):
            agent_type = "prebuilt"
            query = "avg_closure_time('complaint_type')"
            description = "Calculate average closure times using prebuilt analytics"
            
        # For complex statistical questions, use custom analysis
        elif any(pattern in question_lower for pattern in ['correlation', 'regression', 'distribution', 'variance', 'standard deviation']):
            agent_type = "custom"
            query = question  # Pass the full question for custom code generation
            description = "Perform statistical analysis using custom Python code"
            
        return {
            "question": question,
            "complexity": "simple",
            "requires_multi_step": False,
            "steps": [{
                "step_id": "step_1",
                "description": description,
                "agent_type": agent_type,
                "query": query,
                "depends_on": [],
                "output_type": "data"
            }],
            "final_visualization": chart_type
        }
    
    def detect_question_patterns(self, question: str) -> Dict[str, Any]:
        """Detect common question patterns that indicate multi-step analysis"""
        patterns = {
            "top_N_with_condition": {
                "pattern": r"top\s+(\d+).*?(percent|percentage|%)",
                "example": "For the top 5 complaint types, what percent were closed within 3 days?",
                "requires_multi_step": True
            },
            "comparative_analysis": {
                "pattern": r"(compare|versus|vs|difference).*(across|between)",
                "example": "Compare resolution times across boroughs",
                "requires_multi_step": True
            },
            "filtered_ranking": {
                "pattern": r"(which|what).*?(manhattan|brooklyn|queens|bronx|staten).*(fastest|slowest|highest|lowest)",
                "example": "Which Manhattan ZIP codes have the fastest resolution times?",
                "requires_multi_step": True
            },
            "time_trend_analysis": {
                "pattern": r"(trend|over time|monthly|seasonal|pattern)",
                "example": "Show monthly trends for noise complaints",
                "requires_multi_step": True
            }
        }
        
        question_lower = question.lower()
        detected_patterns = []
        
        for pattern_name, pattern_info in patterns.items():
            if re.search(pattern_info["pattern"], question_lower, re.IGNORECASE):
                detected_patterns.append({
                    "name": pattern_name,
                    "requires_multi_step": pattern_info["requires_multi_step"]
                })
        
        return {
            "detected_patterns": detected_patterns,
            "requires_multi_step": any(p["requires_multi_step"] for p in detected_patterns),
            "complexity": "complex" if detected_patterns else "simple"
        }
