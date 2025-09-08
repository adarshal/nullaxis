import sqlite3
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLExecutor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.allowed_tables = ['complaints']
        self.dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 
            'TRUNCATE', 'REPLACE', 'EXEC', 'EXECUTE'
        ]
    
    def validate_sql(self, sql: str) -> bool:
        """Validate SQL query for safety"""
        sql_upper = sql.upper().strip()
        
        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if keyword in sql_upper:
                logger.warning(f"Dangerous keyword '{keyword}' detected in SQL")
                return False
        
        # Must start with SELECT
        if not sql_upper.startswith('SELECT'):
            logger.warning("SQL must start with SELECT")
            return False
        
        # Check for table access
        if not any(table in sql_upper for table in self.allowed_tables):
            logger.warning("SQL must reference allowed tables")
            return False
        
        return True
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        if not self.validate_sql(sql):
            return {
                "error": "Invalid or unsafe SQL query",
                "data": [],
                "chart_type": "error"
            }
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql, conn)
            conn.close()
            
            # Determine chart type based on query structure
            chart_type = self._determine_chart_type(sql, df)
            
            return {
                "data": df.to_dict('records'),
                "chart_type": chart_type,
                "sql": sql,
                "row_count": len(df)
            }
            
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return {
                "error": str(e),
                "data": [],
                "chart_type": "error"
            }
    
    def _determine_chart_type(self, sql: str, df: pd.DataFrame) -> str:
        """Determine appropriate chart type based on SQL and data"""
        sql_upper = sql.upper()
        
        # If query has GROUP BY, likely aggregation
        if 'GROUP BY' in sql_upper:
            # If only 2 columns, likely bar chart
            if len(df.columns) == 2:
                return "bar"
            # If more columns, table view
            else:
                return "table"
        
        # If query has ORDER BY with LIMIT, likely top-k
        if 'ORDER BY' in sql_upper and 'LIMIT' in sql_upper:
            return "bar"
        
        # If single row, single column, likely metric
        if len(df) == 1 and len(df.columns) == 1:
            return "metric"
        
        # If date column present, likely time series
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            return "line"
        
        # Default to table
        return "table"
    
    def get_table_schema(self) -> Dict[str, List[str]]:
        """Get schema information for the complaints table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute("PRAGMA table_info(complaints)")
            columns = cursor.fetchall()
            
            # Get sample data
            cursor.execute("SELECT * FROM complaints LIMIT 5")
            sample_data = cursor.fetchall()
            
            conn.close()
            
            schema = {
                "columns": [col[1] for col in columns],
                "types": [col[2] for col in columns],
                "sample_data": sample_data
            }
            
            return schema
            
        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return {"error": str(e)}
    
    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """Get basic statistics for a column"""
        if column not in self.allowed_tables:
            return {"error": "Invalid column"}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get basic stats
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT({column}) as non_null_count,
                    COUNT(DISTINCT {column}) as unique_count
                FROM complaints
            """)
            
            stats = cursor.fetchone()
            
            # Get top values
            cursor.execute(f"""
                SELECT {column}, COUNT(*) as count
                FROM complaints
                WHERE {column} IS NOT NULL
                GROUP BY {column}
                ORDER BY count DESC
                LIMIT 10
            """)
            
            top_values = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_count": stats[0],
                "non_null_count": stats[1],
                "unique_count": stats[2],
                "null_percentage": round((stats[0] - stats[1]) * 100.0 / stats[0], 2),
                "top_values": top_values
            }
            
        except Exception as e:
            logger.error(f"Failed to get column stats: {e}")
            return {"error": str(e)}
