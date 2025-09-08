import pandas as pd
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrebuiltAnalytics:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return pd.DataFrame()
    
    def get_top_k(self, column: str, k: int = 10) -> Dict[str, Any]:
        """Get top K values by count for a column"""
        query = f"""
        SELECT {column}, COUNT(*) as count
        FROM complaints
        WHERE {column} IS NOT NULL
        GROUP BY {column}
        ORDER BY count DESC
        LIMIT {k}
        """
        
        df = self._execute_query(query)
        return {
            "data": df.to_dict('records'),
            "chart_type": "bar",
            "title": f"Top {k} {column.replace('_', ' ').title()}"
        }
    
    def percent_closed_within_days(self, days: int, group_col: str = None) -> Dict[str, Any]:
        """Calculate percentage of complaints closed within specified days"""
        if group_col:
            query = f"""
            SELECT 
                {group_col},
                COUNT(*) as total,
                SUM(CASE 
                    WHEN closed_date IS NOT NULL 
                    AND JULIANDAY(closed_date) - JULIANDAY(created_date) <= {days} 
                    THEN 1 ELSE 0 
                END) as closed_within_days,
                ROUND(
                    SUM(CASE 
                        WHEN closed_date IS NOT NULL 
                        AND JULIANDAY(closed_date) - JULIANDAY(created_date) <= {days} 
                        THEN 1 ELSE 0 
                    END) * 100.0 / COUNT(*), 2
                ) as percentage
            FROM complaints
            WHERE {group_col} IS NOT NULL
            GROUP BY {group_col}
            ORDER BY total DESC
            LIMIT 10
            """
        else:
            query = f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE 
                    WHEN closed_date IS NOT NULL 
                    AND JULIANDAY(closed_date) - JULIANDAY(created_date) <= {days} 
                    THEN 1 ELSE 0 
                END) as closed_within_days,
                ROUND(
                    SUM(CASE 
                        WHEN closed_date IS NOT NULL 
                        AND JULIANDAY(closed_date) - JULIANDAY(created_date) <= {days} 
                        THEN 1 ELSE 0 
                    END) * 100.0 / COUNT(*), 2
                ) as percentage
            FROM complaints
            """
        
        df = self._execute_query(query)
        return {
            "data": df.to_dict('records'),
            "chart_type": "bar" if group_col else "metric",
            "title": f"Percentage Closed Within {days} Days"
        }
    
    def complaints_by_zip(self, top_k: int = 10) -> Dict[str, Any]:
        """Get complaints by ZIP code"""
        query = f"""
        SELECT 
            incident_zip,
            COUNT(*) as count,
            borough
        FROM complaints
        WHERE incident_zip IS NOT NULL
        GROUP BY incident_zip, borough
        ORDER BY count DESC
        LIMIT {top_k}
        """
        
        df = self._execute_query(query)
        return {
            "data": df.to_dict('records'),
            "chart_type": "bar",
            "title": f"Top {top_k} ZIP Codes by Complaint Count"
        }
    
    def complaints_by_borough(self) -> Dict[str, Any]:
        """Get complaints by borough"""
        query = """
        SELECT 
            borough,
            COUNT(*) as count
        FROM complaints
        WHERE borough IS NOT NULL
        GROUP BY borough
        ORDER BY count DESC
        """
        
        df = self._execute_query(query)
        return {
            "data": df.to_dict('records'),
            "chart_type": "pie",
            "title": "Complaints by Borough"
        }
    
    def geo_validity(self) -> Dict[str, Any]:
        """Check proportion of complaints with valid geocoding"""
        query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE 
                WHEN latitude IS NOT NULL 
                AND longitude IS NOT NULL 
                AND latitude != 0 
                AND longitude != 0 
                THEN 1 ELSE 0 
            END) as geocoded,
            ROUND(
                SUM(CASE 
                    WHEN latitude IS NOT NULL 
                    AND longitude IS NOT NULL 
                    AND latitude != 0 
                    AND longitude != 0 
                    THEN 1 ELSE 0 
                END) * 100.0 / COUNT(*), 2
            ) as percentage
        FROM complaints
        """
        
        df = self._execute_query(query)
        return {
            "data": df.to_dict('records'),
            "chart_type": "metric",
            "title": "Geocoding Validity"
        }
    
    def time_series_trend(self, granularity: str = "month") -> Dict[str, Any]:
        """Get time series trend of complaints"""
        if granularity == "month":
            date_format = "strftime('%Y-%m', created_date)"
        elif granularity == "week":
            date_format = "strftime('%Y-%W', created_date)"
        else:  # day
            date_format = "date(created_date)"
        
        query = f"""
        SELECT 
            {date_format} as period,
            COUNT(*) as count
        FROM complaints
        WHERE created_date IS NOT NULL
        GROUP BY {date_format}
        ORDER BY period
        """
        
        df = self._execute_query(query)
        return {
            "data": df.to_dict('records'),
            "chart_type": "line",
            "title": f"Complaints Trend by {granularity.title()}"
        }
    
    def avg_closure_time(self, group_col: str = None) -> Dict[str, Any]:
        """Calculate average closure time"""
        if group_col:
            query = f"""
            SELECT 
                {group_col},
                COUNT(*) as total,
                ROUND(AVG(JULIANDAY(closed_date) - JULIANDAY(created_date)), 2) as avg_days
            FROM complaints
            WHERE closed_date IS NOT NULL 
            AND created_date IS NOT NULL
            AND {group_col} IS NOT NULL
            GROUP BY {group_col}
            ORDER BY total DESC
            LIMIT 10
            """
        else:
            query = """
            SELECT 
                COUNT(*) as total,
                ROUND(AVG(JULIANDAY(closed_date) - JULIANDAY(created_date)), 2) as avg_days
            FROM complaints
            WHERE closed_date IS NOT NULL 
            AND created_date IS NOT NULL
            """
        
        df = self._execute_query(query)
        return {
            "data": df.to_dict('records'),
            "chart_type": "bar" if group_col else "metric",
            "title": "Average Closure Time (Days)"
        }
    
    def complaints_by_agency(self, top_k: int = 10) -> Dict[str, Any]:
        """Get complaints by agency"""
        query = f"""
        SELECT 
            agency,
            COUNT(*) as count
        FROM complaints
        WHERE agency IS NOT NULL
        GROUP BY agency
        ORDER BY count DESC
        LIMIT {top_k}
        """
        
        df = self._execute_query(query)
        return {
            "data": df.to_dict('records'),
            "chart_type": "bar",
            "title": f"Top {top_k} Agencies by Complaint Count"
        }
    
    def open_vs_closed(self) -> Dict[str, Any]:
        """Get open vs closed complaints distribution"""
        query = """
        SELECT 
            CASE 
                WHEN status = 'Closed' THEN 'Closed'
                ELSE 'Open'
            END as status_group,
            COUNT(*) as count
        FROM complaints
        WHERE status IS NOT NULL
        GROUP BY status_group
        """
        
        df = self._execute_query(query)
        return {
            "data": df.to_dict('records'),
            "chart_type": "pie",
            "title": "Open vs Closed Complaints"
        }
    
    def complaint_distribution_by_descriptor(self, complaint_type: str) -> Dict[str, Any]:
        """Get distribution of descriptors for a specific complaint type"""
        query = """
        SELECT 
            descriptor,
            COUNT(*) as count
        FROM complaints
        WHERE complaint_type = ? 
        AND descriptor IS NOT NULL
        GROUP BY descriptor
        ORDER BY count DESC
        LIMIT 10
        """
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=[complaint_type])
        conn.close()
        
        return {
            "data": df.to_dict('records'),
            "chart_type": "bar",
            "title": f"Descriptors for {complaint_type}"
        }
    
    def get_function_by_name(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """Get prebuilt function result by name"""
        function_map = {
            "get_top_k": self.get_top_k,
            "percent_closed_within_days": self.percent_closed_within_days,
            "complaints_by_zip": self.complaints_by_zip,
            "complaints_by_borough": self.complaints_by_borough,
            "geo_validity": self.geo_validity,
            "time_series_trend": self.time_series_trend,
            "avg_closure_time": self.avg_closure_time,
            "complaints_by_agency": self.complaints_by_agency,
            "open_vs_closed": self.open_vs_closed,
            "complaint_distribution_by_descriptor": self.complaint_distribution_by_descriptor
        }
        
        if function_name in function_map:
            try:
                return function_map[function_name](**kwargs)
            except Exception as e:
                logger.error(f"Function {function_name} failed: {e}")
                return {"error": str(e)}
        else:
            return {"error": f"Function {function_name} not found"}
