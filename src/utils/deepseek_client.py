import openai
import json
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekClient:
    def __init__(self):
        # Default chat model for focused tasks
        self.client = ChatOpenAI(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            temperature=0.1
        )
        
        # Reasoning model for complex planning and multi-step analysis
        self.reasoner = ChatOpenAI(
            model="deepseek-reasoner",
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            temperature=0.1
        )
    
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify user query intent and extract parameters"""
        prompt = f"""
        Analyze this data analytics query and return a JSON response with the following structure:
        
        Query: "{query}"
        
        Return JSON with:
        {{
            "intent": "aggregation|time_series|geospatial|general_stats|custom",
            "columns": ["list", "of", "relevant", "columns"],
            "metric": "count|sum|avg|median|proportion",
            "filters": [{{"column": "name", "operator": "=", "value": "value"}}],
            "group_by": ["column_name"],
            "order_by": "column_name",
            "limit": 10,
            "time_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},
            "confidence": 0.95
        }}
        
        Intent types:
        - aggregation: counting, top-k, averages
        - time_series: trends over time, closure times
        - geospatial: zip codes, boroughs, lat/long analysis
        - general_stats: proportions, validity checks
        - custom: complex analysis requiring custom code
        
        Only return valid JSON, no other text.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "intent": "custom",
                "columns": [],
                "metric": "count",
                "filters": [],
                "confidence": 0.0
            }
    
    def generate_sql(self, intent_data: Dict[str, Any], table_name: str = "complaints") -> str:
        """Generate SQL query from intent data"""
        intent = intent_data.get("intent", "custom")
        columns = intent_data.get("columns", [])
        metric = intent_data.get("metric", "count")
        filters = intent_data.get("filters", [])
        group_by = intent_data.get("group_by", [])
        order_by = intent_data.get("order_by")
        limit = intent_data.get("limit", 10)
        
        prompt = f"""
        Generate a SQL query for the NYC 311 complaints table with these parameters:
        
        Table: {table_name}
        Intent: {intent}
        Columns: {columns}
        Metric: {metric}
        Filters: {filters}
        Group By: {group_by}
        Order By: {order_by}
        Limit: {limit}
        
        Available columns in the table:
        - unique_key, created_date, closed_date, complaint_type, descriptor
        - status, incident_zip, borough, latitude, longitude, agency
        
        Return only the SQL query, no explanations.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            sql = response.choices[0].message.content.strip()
            # Remove any markdown formatting
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.endswith("```"):
                sql = sql[:-3]
            
            return sql.strip()
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return f"SELECT COUNT(*) FROM {table_name}"
    
    def generate_analysis_code(self, query: str, df_name: str = "df") -> str:
        """Generate Python code for custom analysis"""
        prompt = f"""
        Generate Python pandas code to analyze NYC 311 complaints data.
        
        Query: "{query}"
        DataFrame variable name: {df_name}
        
        Available columns: unique_key, created_date, closed_date, complaint_type, descriptor, 
        status, incident_zip, borough, latitude, longitude, agency
        
        Rules:
        1. Only use pandas, no other libraries
        2. Handle missing values appropriately
        3. Return a result variable with the analysis
        4. Include comments explaining the logic
        5. Handle date operations properly
        
        Return only the Python code, no explanations.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            code = response.choices[0].message.content.strip()
            # Remove any markdown formatting
            if code.startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]
            
            return code.strip()
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return f"# Error generating code\nresult = 'Analysis failed'"
    
    def format_analysis_result(self, result: Any, query: str) -> str:
        """Format analysis results into natural language"""
        prompt = f"""
        Convert this data analysis result into a clear, informative summary.
        
        Original Query: "{query}"
        Result: {result}
        
        Provide a natural language explanation that:
        1. Answers the original question clearly
        2. Highlights key insights
        3. Uses appropriate units and formatting
        4. Is concise but informative
        
        Return only the summary text, no additional formatting.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Result formatting failed: {e}")
            return f"Analysis result: {result}"
