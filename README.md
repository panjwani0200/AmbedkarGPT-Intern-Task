# AmbedkarGPT - Intern Task

A Retrieval-Augmented Generation (RAG) based Question Answering system built with LangChain, ChromaDB, and Ollama.

## Overview

This project implements a command-line QA system that:
- Loads a text file (speech.txt)
- Splits text into manageable chunks
- Creates embeddings using HuggingFace models
- Stores embeddings in ChromaDB (local vector database)
- Retrieves relevant context for user queries
- Generates answers using Ollama (Mistral 7B LLM)

## Architecture

The system uses a RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Loading**: Load speech.txt
2. **Text Chunking**: Split text into 500-character chunks with 50-character overlap
3. **Embedding**: Use HuggingFace sentence-transformers to create embeddings
4. **Vector Store**: Store embeddings in ChromaDB (local, no API keys required)
5. **LLM**: Use Ollama with Mistral 7B model for answer generation
6. **Retrieval Chain**: RetrievalQA to fetch relevant chunks and generate answers

## Prerequisites

- Python 3.8+
- Ollama installed with Mistral 7B model
- 8GB+ RAM (recommended)

## Installation

### 1. Install Ollama

Download and install Ollama from https://ollama.ai/

### 2. Install Mistral 7B Model

After installing Ollama, pull the Mistral 7B model:

```bash
ollama pull mistral
```

### 3. Clone Repository and Install Dependencies

```bash
git clone https://github.com/panjwani0200/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

pip install -r requirements.txt
```

## Usage

### 1. Start Ollama Service

```bash
ollama serve
```

### 2. Run the QA System

```bash
python main.py
```

### 3. Ask Questions

```
You: What is the real remedy for caste?
Answer: The real remedy is to destroy the belief in the sanctity of the shastras...
```

Type 'exit' or 'quit' to close the program.

## Technical Details

- **LangChain**: Orchestrates the RAG pipeline
- **ChromaDB**: Local vector database (no internet required)
- **HuggingFace Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Ollama**: LLM inference with Mistral 7B

## Kalpit Pvt Ltd Internship Assignment

This project was developed for Kalpit Pvt Ltd's AI Intern Hiring Assignment.

**Contact**: Kalpik Singh (kalpiksingh2005@gmail.com)
