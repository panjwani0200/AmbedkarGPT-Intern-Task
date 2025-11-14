#!/usr/bin/env python3
"""
AmbedkarGPT - RAG-based Question Answering System
Kalpit Pvt Ltd Internship Assignment
"""

from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os

def initialize_rag_system(speech_file: str = "speech.txt", persist_dir: str = "./chroma_db"):
    """
    Initialize the RAG system with embeddings and vector store.
    
    Args:
        speech_file: Path to the speech text file
        persist_dir: Directory to persist ChromaDB
        
    Returns:
        RetrievalQA chain ready for querying
    """
    
    print("[1/5] Loading text file...")
    loader = TextLoader(speech_file)
    documents = loader.load()
    
    print("[2/5] Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    print(f"    Created {len(docs)} chunks from the document")
    
    print("[3/5] Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("[4/5] Storing embeddings in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print("[5/5] Initializing Ollama LLM and creating QA chain...")
    llm = Ollama(model="mistral")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    
    print("\n✓ RAG system initialized successfully!\n")
    return qa_chain

def main():
    """Main function to run the QA system."""
    
    if not os.path.exists("speech.txt"):
        print("Error: speech.txt not found!")
        return
    
    qa_chain = initialize_rag_system()
    
    print("="*60)
    print("AmbedkarGPT - Question Answering System")
    print("="*60)
    print("Type your questions about the provided text.")
    print("Type 'exit' or 'quit' to end the program.\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("Thank you for using AmbedkarGPT. Goodbye!")
            break
        
        if not question:
            continue
        
        print("\nProcessing your question...\n")
        result = qa_chain.run(question)
        print(f"Answer: {result}\n")
        print("-"*60 + "\n")

if __name__ == "__main__":
    main()
