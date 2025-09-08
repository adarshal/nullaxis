#!/usr/bin/env python3
"""
Setup script for existing NYC 311 data
Uses the existing CSV file: 311_Service_Requests_from_2010_to_Present.csv
"""
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import sqlite3
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NYC311Setup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.csv_path = self.project_root / "311_Service_Requests_from_2010_to_Present.csv"
        self.db_path = self.data_dir / "nyc311.db"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
    
    # def install_dependencies(self):
    #     """Install required Python packages"""
    #     print("📦 Installing dependencies...")
    #     try:
    #         subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    #         print("✅ Dependencies installed successfully")
    #         return True
    #     except subprocess.CalledProcessError as e:
    #         print(f"❌ Failed to install dependencies: {e}")
    #         return False
    
    def check_data_file(self):
        """Check if the data file exists"""
        print("📁 Checking data file...")
        
        if not self.csv_path.exists():
            print(f"❌ Data file not found: {self.csv_path}")
            print("Please make sure '311_Service_Requests_from_2010_to_Present.csv' is in the root folder")
            return False
        
        print(f"✅ Found data file: {self.csv_path}")
        return True
    
    def create_database(self):
        """Create SQLite database from existing CSV data"""
        print("🗄️  Creating SQLite database...")
        
        try:
            # Read CSV data and process for analytics (works with any date range)
            print("Reading and processing NYC 311 CSV data...")
            
            # Read CSV in chunks to avoid memory issues
            chunks = pd.read_csv(self.csv_path, chunksize=100_000, low_memory=False)
            frames = []
            total_processed = 0
            date_range_info = None
            
            for chunk_num, ch in enumerate(chunks):
                total_processed += len(ch)
                print(f"Processing chunk {chunk_num + 1} ({len(ch):,} rows, {total_processed:,} total processed)")
                
                # Clean column names once per chunk
                ch.columns = ch.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.lower()
                
                # Parse dates safely with explicit format handling
                for col in ['created_date', 'closed_date']:
                    if col in ch.columns:
                        ch[col] = pd.to_datetime(ch[col], errors='coerce')
                
                # Get date range info for the first chunk
                if date_range_info is None and 'created_date' in ch.columns:
                    valid_dates = ch['created_date'].dropna()
                    if not valid_dates.empty:
                        date_range_info = {
                            'min': valid_dates.min(),
                            'max': valid_dates.max()
                        }
                        print(f"Data range: {date_range_info['min']} to {date_range_info['max']}")
                
                # For testing: If you want to filter to recent years, uncomment the next lines:
                # Filtering to 2015+ for this dataset (adjust as needed for actual 2020+ data)
                # if 'created_date' in ch.columns:
                #     ch = ch[ch['created_date'] >= '2015-01-01']
                #     print(f"After date filtering: {len(ch):,} records")
                
                # Skip empty chunks
                if ch.empty:
                    continue
                    
                # Add derived columns for analytics
                if {'created_date', 'closed_date'}.issubset(ch.columns):
                    # Closed within 3 days (for sample question: "what percent were closed within 3 days?")
                    closure_days = (ch['closed_date'] - ch['created_date']).dt.days
                    ch['closed_within_3_days'] = (closure_days <= 3) & (closure_days >= 0)
                    print(f"Added closed_within_3_days column")
                else:
                    ch['closed_within_3_days'] = pd.NA

                if {'latitude', 'longitude'}.issubset(ch.columns):
                    # Geocoded status (for sample question: "proportion with valid lat/lng")
                    ch['is_geocoded'] = (ch['latitude'].notna() & ch['longitude'].notna() & 
                                       (ch['latitude'] != 0) & (ch['longitude'] != 0))
                    print(f"Added is_geocoded column")
                else:
                    ch['is_geocoded'] = pd.NA

                frames.append(ch)

            if not frames:
                raise ValueError("No data found after processing. Check your CSV file format and content.")
                
            df = pd.concat(frames, ignore_index=True)
            print(f"Final dataset: {len(df):,} records")
            print(f"Columns: {list(df.columns)}")
            
            # Verify derived columns
            if 'closed_within_3_days' in df.columns:
                closed_3d_count = df['closed_within_3_days'].sum()
                print(f"Records closed within 3 days: {closed_3d_count:,}")
            
            if 'is_geocoded' in df.columns:
                geocoded_count = df['is_geocoded'].sum()
                print(f"Records with valid geocoding: {geocoded_count:,}")
            
            # Create database
            print("Creating database...")
            conn = sqlite3.connect(self.db_path)
            df.to_sql('complaints', conn, if_exists='replace', index=False)
            
            # Create indexes for performance
            print("Creating indexes...")
            cursor = conn.cursor()
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_complaints_type ON complaints(complaint_type)",
                "CREATE INDEX IF NOT EXISTS idx_complaints_zip ON complaints(incident_zip)",
                "CREATE INDEX IF NOT EXISTS idx_complaints_date ON complaints(created_date)",
                "CREATE INDEX IF NOT EXISTS idx_complaints_borough ON complaints(borough)",
                "CREATE INDEX IF NOT EXISTS idx_complaints_status ON complaints(status)",
                "CREATE INDEX IF NOT EXISTS idx_closed_3days ON complaints(closed_within_3_days)",
                "CREATE INDEX IF NOT EXISTS idx_geocoded ON complaints(is_geocoded)",
                "CREATE INDEX IF NOT EXISTS idx_agency ON complaints(agency)",
                "CREATE INDEX IF NOT EXISTS idx_closed_date ON complaints(closed_date)"
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as e:
                    print(f"Warning: Could not create index: {e}")
            
            conn.commit()
            conn.close()
            
            print(f"✅ Database created at {self.db_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create database: {e}")
            return False
    
    def get_data_summary(self):
        """Get summary statistics of the data"""
        print("📊 Generating data summary...")
        
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
            
            print(f"📈 Data Summary:")
            print(f"   Total records: {total_records:,}")
            if date_range[0] and date_range[1]:
                print(f"   Date range: {date_range[0]} to {date_range[1]}")
            print(f"   Top complaint types:")
            for i, (complaint_type, count) in enumerate(top_complaints, 1):
                print(f"     {i}. {complaint_type}: {count:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate summary: {e}")
            return False
    
    def create_env_file(self):
        """Create .env file template"""
        print("📝 Creating .env file template...")
        
        env_content = """# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Database Configuration
DATABASE_PATH=data/nyc311.db
CSV_PATH=311_Service_Requests_from_2010_to_Present.csv
"""
        
        env_path = self.project_root / ".env"
        if not env_path.exists():
            with open(env_path, 'w') as f:
                f.write(env_content)
            print("✅ .env file created")
            print("⚠️  Please edit .env and add your DeepSeek API key")
        else:
            print("✅ .env file already exists")
        
        return True
    
    def run_setup(self):
        """Run complete setup process"""
        print("🚀 NYC 311 Data Analytics Agent Setup")
        print("=" * 50)
        print("Using existing data file: 311_Service_Requests_from_2010_to_Present.csv")
        print("=" * 50)
        
        steps = [
            # ("Installing dependencies", self.install_dependencies),
            ("Checking data file", self.check_data_file),
            ("Creating .env file", self.create_env_file),
            ("Creating database", self.create_database),
            ("Generating summary", self.get_data_summary)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if not step_func():
                print(f"❌ Setup failed at: {step_name}")
                return False
        
        print("\n" + "=" * 50)
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your DeepSeek API key")
        print("2. Run: streamlit run app.py")
        print("\nOr test the setup with: python test_setup.py")
        
        return True

def main():
    setup = NYC311Setup()
    success = setup.run_setup()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
