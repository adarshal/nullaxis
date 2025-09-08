import pandas as pd
import sqlite3
import os
import requests
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, csv_path: str, db_path: str):
        self.csv_path = csv_path
        self.db_path = db_path
        
    def download_nyc_data(self, url: str = None) -> bool:
        """Check if NYC 311 data exists"""
        if os.path.exists(self.csv_path):
            logger.info(f"Data already exists at {self.csv_path}")
            return True
            
        if not url:
            # Default NYC 311 data URL - you may need to update this
            url = "https://data.cityofnewyork.us/api/views/erm2-nwe9/rows.csv?accessType=DOWNLOAD"
        
        try:
            logger.info("Downloading NYC 311 data...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            
            with open(self.csv_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Data downloaded successfully to {self.csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            return False
    
    def create_database(self) -> bool:
        """Create SQLite database and load data"""
        try:
            # Create database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to SQLite database
            conn = sqlite3.connect(self.db_path)
            
            # Read CSV data
            logger.info("Loading CSV data into database...")
            df = pd.read_csv(self.csv_path)
            
            # Clean column names (remove spaces, special characters)
            df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.lower()
            
            # Convert date columns
            date_columns = ['created_date', 'closed_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Load data into SQLite
            df.to_sql('complaints', conn, if_exists='replace', index=False)
            
            # Create indexes for performance
            logger.info("Creating database indexes...")
            cursor = conn.cursor()
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_complaints_type ON complaints(complaint_type)",
                "CREATE INDEX IF NOT EXISTS idx_complaints_zip ON complaints(incident_zip)",
                "CREATE INDEX IF NOT EXISTS idx_complaints_date ON complaints(created_date)",
                "CREATE INDEX IF NOT EXISTS idx_complaints_borough ON complaints(borough)",
                "CREATE INDEX IF NOT EXISTS idx_complaints_status ON complaints(status)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database created successfully at {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return False
    
    def get_data_summary(self) -> dict:
        """Get basic summary statistics of the data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            summary = {}
            
            # Total records
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM complaints")
            summary['total_records'] = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("SELECT MIN(created_date), MAX(created_date) FROM complaints WHERE created_date IS NOT NULL")
            date_range = cursor.fetchone()
            summary['date_range'] = {
                'start': date_range[0],
                'end': date_range[1]
            }
            
            # Top complaint types
            cursor.execute("""
                SELECT complaint_type, COUNT(*) as count 
                FROM complaints 
                WHERE complaint_type IS NOT NULL 
                GROUP BY complaint_type 
                ORDER BY count DESC 
                LIMIT 5
            """)
            summary['top_complaint_types'] = cursor.fetchall()
            
            # Borough distribution
            cursor.execute("""
                SELECT borough, COUNT(*) as count 
                FROM complaints 
                WHERE borough IS NOT NULL 
                GROUP BY borough 
                ORDER BY count DESC
            """)
            summary['borough_distribution'] = cursor.fetchall()
            
            conn.close()
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return {}
    
    def setup_data(self) -> bool:
        """Complete data setup process"""
        logger.info("Starting data setup process...")
        
        # Download data if needed
        if not self.download_nyc_data():
            return False
        
        # Create database
        if not self.create_database():
            return False
        
        # Get summary
        summary = self.get_data_summary()
        logger.info(f"Data setup complete. Summary: {summary}")
        
        return True
