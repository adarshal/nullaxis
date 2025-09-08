#!/usr/bin/env python3
"""
Download NYC 311 data and setup database
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.data_processor import DataProcessor
from config import DATABASE_PATH, CSV_PATH

def main():
    print("NYC 311 Data Setup")
    print("=" * 30)
    
    # Create data processor
    processor = DataProcessor(CSV_PATH, DATABASE_PATH)
    
    # Download and setup data
    print("Downloading NYC 311 data...")
    if processor.setup_data():
        print("✅ Data setup completed successfully!")
        
        # Show summary
        summary = processor.get_data_summary()
        if summary:
            print(f"\n📊 Data Summary:")
            print(f"Total records: {summary.get('total_records', 0):,}")
            
            if 'date_range' in summary:
                print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
            
            if 'top_complaint_types' in summary:
                print(f"\nTop complaint types:")
                for i, (complaint_type, count) in enumerate(summary['top_complaint_types'][:5], 1):
                    print(f"{i}. {complaint_type}: {count:,}")
    else:
        print("❌ Data setup failed!")
        return 1
    
    print(f"\nDatabase created at: {DATABASE_PATH}")
    print("You can now run the app with: streamlit run app.py")
    
    return 0

if __name__ == "__main__":
    exit(main())
