import sys
import os

# Ensure we can import from the directory
sys.path.append(os.getcwd())

try:
    print("Testing imports...")
    import config
    import rag_engine
    import utils
    print("Imports successful.")

    print("Testing config...")
    print(f"Ollama URL: {config.OLLAMA_URL}")
    
    # We can't easily test Streamlit cached functions without a running Streamlit app or mocking st.cache_resource.
    # But we can test non-cached functions.
    
    print("Testing text utils...")
    short = rag_engine.ellipsize("This is a test string", n=10)
    print(f"Ellipsize: {short}")
    
    print("Testing routing...")
    route_numeric = rag_engine.route("What is the score for hospital X?")
    print(f"Route 'score': {route_numeric}") # Should be 'facts'
    
    route_interp = rag_engine.route("How can they improve?")
    print(f"Route 'improve': {route_interp}") # Should be 'summaries'

    print("Refactor verification complete.")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
