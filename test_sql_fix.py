
#!/usr/bin/env python3
"""
Test script to verify SQL execution fixes
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.data_processor import DataProcessor
from src.agents.sql_agent import NYC311SQLAgent
from src.utils.deepseek_client import DeepSeekClient
from config import DATABASE_PATH, CSV_PATH, DEEPSEEK_API_KEY

def test_sql_execution():
    """Test that SQL queries are actually executed"""
    print("🧪 Testing SQL Execution Fixes")
    print("=" * 40)
    
    # Check if API key is set
    if not DEEPSEEK_API_KEY:
        print("❌ DEEPSEEK_API_KEY not set. Please set it in .env file")
        return False
    
    # Check if database exists
    if not os.path.exists(DATABASE_PATH):
        print("❌ Database not found. Please run setup first:")
        print("python setup_existing_data.py")
        return False
    
    try:
        # Initialize components
        print("Initializing components...")
        deepseek_client = DeepSeekClient()
        sql_agent = NYC311SQLAgent(DATABASE_PATH, deepseek_client.client)
        
        # Test questions
        test_questions = [
            "What are the top 5 complaint types?",
            "How many complaints are there in total?",
            "Which borough has the most complaints?"
        ]
        
        print("\nTesting SQL queries...")
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 50)
            
            try:
                response = sql_agent.run(question)
                print(f"Response: {response[:300]}...")
                
                # Check if response contains SQL query
                if "SQL Query:" in response or "Query executed successfully" in response:
                    print("✅ SQL query was executed!")
                else:
                    print("⚠️  No SQL execution detected in response")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Show SQL log
        print("\n" + "=" * 50)
        print("📋 SQL Query Log:")
        print("=" * 50)
        
        if os.path.exists('sql_commands.log'):
            with open('sql_commands.log', 'r') as f:
                log_content = f.read()
                print(log_content[-2000:])  # Last 2000 characters
        else:
            print("No SQL log file found")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_sql_execution()
    if success:
        print("\n✅ SQL execution test completed!")
        print("Check the responses above to see if SQL queries were executed.")
    else:
        print("\n❌ SQL execution test failed!")
        sys.exit(1)
