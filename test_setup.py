#!/usr/bin/env python3
"""
Test script to verify the setup
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.utils.data_processor import DataProcessor
        print("✅ DataProcessor imported successfully")
    except Exception as e:
        print(f"❌ DataProcessor import failed: {e}")
        return False
    
    try:
        from src.utils.deepseek_client import DeepSeekClient
        print("✅ DeepSeekClient imported successfully")
    except Exception as e:
        print(f"❌ DeepSeekClient import failed: {e}")
        return False
    
    try:
        from src.agents.sql_agent import NYC311SQLAgent
        print("✅ NYC311SQLAgent imported successfully")
    except Exception as e:
        print(f"❌ NYC311SQLAgent import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    
    try:
        from config import DEEPSEEK_API_KEY, DATABASE_PATH, CSV_PATH
        print("✅ Config imported successfully")
        
        if not DEEPSEEK_API_KEY:
            print("⚠️  DEEPSEEK_API_KEY not set - you'll need to set this to run the app")
        else:
            print("✅ DEEPSEEK_API_KEY is set")
        
        print(f"Database path: {DATABASE_PATH}")
        print(f"CSV path: {CSV_PATH}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_dependencies():
    """Test if required packages are installed"""
    print("\nTesting dependencies...")
    
    required_packages = [
        'streamlit',
        'langgraph',
        'langchain',
        'langchain_community',
        'langchain_openai',
        'pandas',
        'plotly',
        'requests',
        'python_dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def main():
    print("NYC 311 Data Analytics Agent - Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test dependencies
    if not test_dependencies():
        all_passed = False
    
    # Test config
    if not test_config():
        all_passed = False
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! You're ready to run the app.")
        print("\nNext steps:")
        print("1. Set your DEEPSEEK_API_KEY in .env file")
        print("2. Run: python download_data.py")
        print("3. Run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
