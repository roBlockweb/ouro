"""Show the Python path when running this script"""
import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
print(f"Python version: {sys.version}")

try:
    import llama_cpp
    print(f"Found llama_cpp at: {llama_cpp.__file__}")
except ImportError as e:
    print(f"Could not import llama_cpp: {e}")