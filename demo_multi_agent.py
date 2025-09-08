#!/usr/bin/env python3
"""
Demo script to showcase the multi-agent system capabilities
"""
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.analysis_orchestrator import AnalysisOrchestrator
from src.agents.sql_agent import NYC311SQLAgent
from src.utils.deepseek_client import DeepSeekClient
from config import DATABASE_PATH, DEEPSEEK_API_KEY

def demo_multi_agent_system():
    """Demonstrate multi-agent system with different question types"""
    print("🤖 NYC 311 MULTI-AGENT ANALYTICS SYSTEM DEMO")
    print("=" * 60)
    
    if not DEEPSEEK_API_KEY:
        print("❌ Please set DEEPSEEK_API_KEY in your .env file")
        return False
    
    try:
        # Initialize the multi-agent system
        print("🚀 Initializing multi-agent system...")
        deepseek_client = DeepSeekClient()
        sql_agent = NYC311SQLAgent(DATABASE_PATH, deepseek_client.client)
        orchestrator = AnalysisOrchestrator(deepseek_client, sql_agent, DATABASE_PATH)
        print("✅ Multi-agent system initialized successfully!\n")
        
        # Demo questions
        demo_questions = [
            {
                "question": "What are the top 5 complaint types?",
                "type": "Simple Analysis",
                "description": "Single-step aggregation query"
            },
            {
                "question": "For the top 5 complaint types, what percent were closed within 3 days?", 
                "type": "Multi-Step Analysis",
                "description": "Complex query requiring multiple coordinated steps"
            }
        ]
        
        for i, demo in enumerate(demo_questions, 1):
            print(f"🔍 DEMO {i}: {demo['type']}")
            print(f"Question: \"{demo['question']}\"")
            print(f"Description: {demo['description']}")
            print("-" * 60)
            
            try:
                result = orchestrator.execute_analysis(demo["question"])
                
                if result["success"]:
                    print(f"✅ Analysis completed successfully!")
                    print(f"📊 Data records found: {len(result.get('data', []))}")
                    print(f"📈 Chart type: {result['chart_type']}")
                    print(f"🔧 Execution steps: {len(result.get('execution_steps', []))}")
                    print(f"\n💬 Natural Language Response:")
                    print(f"   {result['natural_language_response'][:200]}...")
                    
                    if result.get("execution_steps"):
                        print(f"\n🤖 Execution Steps:")
                        for j, step in enumerate(result["execution_steps"], 1):
                            status = "✅ Success" if step["success"] else "❌ Failed"
                            print(f"   {j}. {step.get('step_id', 'Unknown')}: {status}")
                            if step["success"] and "row_count" in step.get("metadata", {}):
                                print(f"      → Processed {step['metadata']['row_count']} rows")
                    
                else:
                    print(f"❌ Analysis failed: {result['natural_language_response']}")
                
            except Exception as e:
                print(f"❌ Demo {i} failed with error: {e}")
            
            print("\n" + "=" * 60 + "\n")
        
        print("🎉 MULTI-AGENT SYSTEM DEMO COMPLETED!")
        print("\n🚀 **Key Features Demonstrated:**")
        print("✅ Automatic complexity detection (simple vs multi-step)")
        print("✅ Query Planner using deepseek-reasoner for complex analysis")
        print("✅ Analysis Orchestrator coordinating multiple agents")
        print("✅ Structured data output ready for visualization")
        print("✅ Transparent execution step tracking")
        print("✅ Natural language response generation")
        
        print(f"\n💻 **Ready to use!** Start the app with: streamlit run app.py")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_multi_agent_system()
    exit(0 if success else 1)
