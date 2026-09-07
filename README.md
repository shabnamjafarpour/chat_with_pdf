# AradBook RAG Assistant

A PDF-based RAG assistant built with LangChain, FAISS, OpenRouter, and Gradio.

## Current Architecture

PDF
↓
PyPDFLoader
↓
RecursiveCharacterTextSplitter
↓
OpenAI Embeddings
↓
FAISS
↓
Similarity Search
↓
OpenRouter LLM
↓
Gradio Chat Interface

## Tech Stack

- Python
- LangChain
- FAISS
- OpenRouter
- OpenAI Embeddings
- Gradio

## Current Features

- Load PDF documents
- Split documents into chunks
- Generate embeddings
- Store embeddings in FAISS
- Retrieve relevant chunks
- Generate answers using an LLM
- Chat interface with Gradio

## Future Improvements

- Chunkless RAG
- Vectorless Retrieval
- Hybrid Retrieval
- Reranking
- LangGraph
- Evaluation with LangSmith