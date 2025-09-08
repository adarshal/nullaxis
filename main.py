import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.data_processor import DataProcessor
from src.agents.sql_agent import NYC311SQLAgent
from src.utils.deepseek_client import DeepSeekClient
from config import DATABASE_PATH, CSV_PATH, DEEPSEEK_API_KEY
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_data():
    """Setup the NYC 311 data"""
    processor = DataProcessor(CSV_PATH, DATABASE_PATH)
    
    if not processor.setup_data():
        logger.error("Failed to setup data")
        return False
    
    logger.info("Data setup completed successfully")
    return True

def main():
    """Main application entry point"""
    print("NYC 311 Data Analytics Agent")
    print("=" * 40)
    
    # Check if API key is set
    if not DEEPSEEK_API_KEY:
        print("❌ Please set DEEPSEEK_API_KEY in your environment or .env file")
        return
    
    # Setup data
    print("📊 Setting up data...")
    if not setup_data():
        print("❌ Data setup failed. Please check the error messages above.")
        return
    
    # Initialize components
    print("🔧 Initializing components...")
    
    try:
        # Initialize DeepSeek client
        deepseek_client = DeepSeekClient()
        
        # Initialize SQL agent
        sql_agent = NYC311SQLAgent(DATABASE_PATH, deepseek_client.client)
        
        print("✅ Setup complete! You can now run the Streamlit app with: streamlit run app.py")
        
        # Interactive mode for testing
        print("\n💬 Interactive mode (type 'quit' to exit):")
        while True:
            question = input("\nAsk a question about NYC 311 data: ")
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            try:
                print("🤔 Analyzing your question...")
                answer = sql_agent.run(question)
                print(f"\n📋 Answer: {answer}")
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Try rephrasing your question or check if the data contains the information you're looking for.")
                
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return

if __name__ == "__main__":
    main()
