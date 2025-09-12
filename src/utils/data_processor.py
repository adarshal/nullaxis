import sqlite3
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, csv_path: str, db_path: str):
        self.csv_path = csv_path
        self.db_path = db_path
        
    def get_data_summary(self) -> Dict[str, Any]:
        """Get basic summary statistics of the data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM complaints")
            total_records = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("SELECT MIN(created_date), MAX(created_date) FROM complaints WHERE created_date IS NOT NULL")
            date_range = cursor.fetchone()
            
            # Top complaint types
            cursor.execute("""
                SELECT complaint_type, COUNT(*) as count 
                FROM complaints 
                WHERE complaint_type IS NOT NULL 
                GROUP BY complaint_type 
                ORDER BY count DESC 
                LIMIT 5
            """)
            top_complaints = cursor.fetchall()
            
            # Borough distribution
            cursor.execute("""
                SELECT borough, COUNT(*) as count 
                FROM complaints 
                WHERE borough IS NOT NULL 
                GROUP BY borough 
                ORDER BY count DESC
            """)
            borough_dist = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_records': total_records,
                'date_range': {'start': date_range[0], 'end': date_range[1]},
                'top_complaint_types': top_complaints,
                'borough_distribution': borough_dist
            }
            
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return {}