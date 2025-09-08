import pandas as pd
import sqlite3
import logging
import traceback
from typing import Dict, Any, Optional
import re
import sys
from io import StringIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeWriter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.allowed_modules = ['pandas', 'numpy', 'datetime', 'math', 'statistics']
        self.max_execution_time = 30  # seconds
    
    def execute_custom_code(self, code: str, query: str) -> Dict[str, Any]:
        """Execute custom Python code for data analysis"""
        try:
            # Validate code safety
            if not self._validate_code(code):
                return {
                    "error": "Code contains unsafe operations",
                    "result": None,
                    "chart_type": "error"
                }
            
            # Load data
            df = self._load_data()
            if df.empty:
                return {
                    "error": "Failed to load data",
                    "result": None,
                    "chart_type": "error"
                }
            
            # Create execution environment
            exec_globals = {
                'pd': pd,
                'df': df,
                'result': None,
                'print': self._safe_print
            }
            
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                # Execute code
                exec(code, exec_globals)
                
                # Get result
                result = exec_globals.get('result', 'No result variable set')
                output = captured_output.getvalue()
                
                # Determine chart type
                chart_type = self._determine_chart_type_from_result(result)
                
                return {
                    "result": result,
                    "output": output,
                    "chart_type": chart_type,
                    "code": code
                }
                
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "result": None,
                "chart_type": "error"
            }
    
    def _validate_code(self, code: str) -> bool:
        """Validate code for safety"""
        # Check for dangerous operations
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'import\s+shutil',
            r'__import__',
            r'eval\(',
            r'exec\(',
            r'open\(',
            r'file\(',
            r'input\(',
            r'raw_input\(',
            r'compile\(',
            r'reload\(',
            r'execfile\(',
            r'getattr\(',
            r'setattr\(',
            r'delattr\(',
            r'hasattr\(',
            r'globals\(',
            r'locals\(',
            r'vars\(',
            r'dir\(',
            r'help\(',
            r'quit\(',
            r'exit\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected: {pattern}")
                return False
        
        # Check for allowed imports only
        import_lines = re.findall(r'import\s+(\w+)', code)
        for module in import_lines:
            if module not in self.allowed_modules:
                logger.warning(f"Disallowed import: {module}")
                return False
        
        return True
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM complaints", conn)
            conn.close()
            
            # Convert date columns
            date_columns = ['created_date', 'closed_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return pd.DataFrame()
    
    def _determine_chart_type_from_result(self, result: Any) -> str:
        """Determine chart type based on result"""
        if isinstance(result, pd.DataFrame):
            if len(result) == 1 and len(result.columns) == 1:
                return "metric"
            elif len(result.columns) == 2:
                return "bar"
            else:
                return "table"
        elif isinstance(result, (int, float)):
            return "metric"
        elif isinstance(result, (list, tuple)) and len(result) <= 2:
            return "bar"
        else:
            return "table"
    
    def _safe_print(self, *args, **kwargs):
        """Safe print function that doesn't interfere with execution"""
        pass
    
    def generate_analysis_template(self, query: str) -> str:
        """Generate a template for custom analysis"""
        template = f'''
# Custom Analysis for: "{query}"
# Available data in 'df' DataFrame

# Example analysis structure:
# 1. Filter data if needed
# filtered_df = df[df['column'] == 'value']

# 2. Perform analysis
# result = df.groupby('column').size().sort_values(ascending=False)

# 3. Set result variable
# result = your_analysis_result

# Your analysis code here:
result = "Please implement your analysis"
'''
        return template
    
    def suggest_analysis_approach(self, query: str) -> str:
        """Suggest analysis approach based on query"""
        query_lower = query.lower()
        
        if 'top' in query_lower and 'complaint' in query_lower:
            return '''
# Suggested approach for top complaints:
result = df['complaint_type'].value_counts().head(10)
'''
        elif 'percent' in query_lower and 'closed' in query_lower:
            return '''
# Suggested approach for closure percentage:
df['closure_days'] = (df['closed_date'] - df['created_date']).dt.days
closed_within_3_days = (df['closure_days'] <= 3).sum()
total_closed = df['closed_date'].notna().sum()
res
'''