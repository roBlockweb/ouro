from setuptools import setup, find_packages

setup(
    name="ouro",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.28.0",
        "torch>=1.11.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "langchain>=0.0.267",
        "langchain-community>=0.0.6",
        "langchain-text-splitters>=0.0.1",
        "langchain-huggingface>=0.0.1",
        "huggingface_hub>=0.16.4",
        "tqdm>=4.65.0",
        "numpy>=1.24.3",
        "pypdf>=3.12.1",
        "rich>=13.4.2",
        "accelerate>=0.26.0", 
        "safetensors>=0.4.0",
    ],
    python_requires=">=3.9",
    description="Ouro: Offline RAG System",
    author="roBlock",
    author_email="",
    url="https://github.com/roBlock/ouro",
)