"""Check if all required modules can be imported."""
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

# Try importing all required modules
modules = [
    "llama_cpp",
    "faiss",
    "numpy",
    "sentence_transformers",
    "selenium",
    "bs4",
    "requests",
    "logging",
    "time",
    "random",
    "datetime"
]

print("\nChecking modules:")
for module in modules:
    try:
        __import__(module)
        print(f"✅ {module}: OK")
    except ImportError as e:
        print(f"❌ {module}: FAILED - {str(e)}")

print("\nDone checking modules.")