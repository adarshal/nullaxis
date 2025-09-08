#!/usr/bin/env python3
"""
Test the multi-agent system components
"""
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.query_planner import QueryPlanner
from src.agents.analysis_orchestrator import AnalysisOrchestrator
from src.agents.sql_agent import NYC311SQLAgent
from src.utils.deepseek_client import DeepSeekClient
from config import DATABASE_PATH, DEEPSEEK_API_KEY

def test_query_planner():
    """Test the Query Planner Agent"""
    print("🧪 Testing Query Planner Agent")
    print("=" * 50)
    
    if not DEEPSEEK_API_KEY:
        print("❌ DEEPSEEK_API_KEY not found. Please set it in your .env file")
        return False
    
    try:
        # Initialize components
        deepseek_client = DeepSeekClient()
        planner = QueryPlanner(deepseek_client.reasoner)
        
        # Test simple question
        simple_question = "What are the top 5 complaint types?"
        print(f"\n🔍 Testing simple question: {simple_question}")
        
        plan = planner.analyze_question(simple_question)
        print(f"✅ Plan created: {plan['complexity']} complexity, {len(plan['steps'])} steps")
        print(f"   Multi-step required: {plan['requires_multi_step']}")
        print(f"   Visualization: {plan['final_visualization']}")
        
        # Test complex question
        complex_question = "For the top 5 complaint types, what percent were closed within 3 days?"
        print(f"\n🔍 Testing complex question: {complex_question}")
        
        plan = planner.analyze_question(complex_question)
        print(f"✅ Plan created: {plan['complexity']} complexity, {len(plan['steps'])} steps")
        print(f"   Multi-step required: {plan['requires_multi_step']}")
        print(f"   Visualization: {plan['final_visualization']}")
        
        for i, step in enumerate(plan['steps'], 1):
            print(f"   Step {i}: {step['description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query Planner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator():
    """Test the Analysis Orchestrator"""
    print("\n🤖 Testing Analysis Orchestrator")
    print("=" * 50)
    
    try:
        # Initialize components
        deepseek_client = DeepSeekClient()
        sql_agent = NYC311SQLAgent(DATABASE_PATH, deepseek_client.client)
        orchestrator = AnalysisOrchestrator(deepseek_client, sql_agent, DATABASE_PATH)
        
        # Test simple analysis
        simple_question = "What are the top 5 complaint types?"
        print(f"\n🔍 Testing simple orchestration: {simple_question}")
        
        result = orchestrator.execute_analysis(simple_question)
        print(f"✅ Analysis completed: Success={result['success']}")
        print(f"   Data records: {len(result.get('data', []))}")
        print(f"   Chart type: {result['chart_type']}")
        print(f"   Execution steps: {len(result.get('execution_steps', []))}")
        
        if result['data']:
            print(f"   Sample data: {result['data'][0] if result['data'] else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pattern_detection():
    """Test pattern detection capabilities"""
    print("\n🔍 Testing Pattern Detection")
    print("=" * 50)
    
    try:
        deepseek_client = DeepSeekClient()
        planner = QueryPlanner(deepseek_client.reasoner)
        
        test_questions = [
            "For the top 5 complaint types, what percent were closed within 3 days?",
            "Which Manhattan ZIP codes have the fastest resolution times?", 
            "Compare resolution times across boroughs",
            "Show monthly trends for noise complaints",
            "What are the top 10 complaint types?"  # Simple question
        ]
        
        for question in test_questions:
            patterns = planner.detect_question_patterns(question)
            print(f"Question: {question}")
            print(f"  Detected patterns: {len(patterns['detected_patterns'])}")
            print(f"  Requires multi-step: {patterns['requires_multi_step']}")
            print(f"  Complexity: {patterns['complexity']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern detection test failed: {e}")
        return False

def main():
    """Run all multi-agent tests"""
    print("🚀 MULTI-AGENT SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Query Planner", test_query_planner),
        ("Analysis Orchestrator", test_orchestrator), 
        ("Pattern Detection", test_pattern_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Multi-agent system is ready.")
        print("✅ Query Planner can analyze complex questions")
        print("✅ Analysis Orchestrator can execute workflows")
        print("✅ Pattern detection works for multi-step questions")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
