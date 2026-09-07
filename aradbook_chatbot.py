import os

import gradio as gr
import arabic_reshaper
from bidi.algorithm import get_display

from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()


def load_model(model_name: str = "openai/gpt-4o-mini"):
    llm_model = ChatOpenRouter(
        model=model_name,
        max_tokens=300,
    )

    return llm_model


def text_chunker(documents: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks


def load_document(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents


def load_embedding_model(embedding_model):
    embedding_model = OpenAIEmbeddings(
        model=embedding_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    return embedding_model


def chat(message, history, vector_store, llm_model):
    """
    This function is called by Gradio whenever
    the user sends a new message.
    """

    # 1. Search relevant documents
    results = vector_store.similarity_search(
        message,
        k=4  #search for top k in vector database
    )

    # 2. Build context
    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    # 3. Build prompt
    prompt = f"""
        You are a helpful assistant.

        Answer the user's question based only on the provided context.

        If the answer cannot be found in the context, say:
        "I don't know based on the provided document."

        Context:
        {context}

        Question:
        {message}

        Answer:
     """

    # 4. Send prompt to LLM
    response = llm_model.invoke(prompt)

    # 5. Return answer to Gradio
    return response.content


def main():

    FILE_PATH = r"D:\AI_REPOSITORY_CACHE_MI\projects\AradBook\docs\external_data.pdf"

    EMBEDDING_MODEL_NAME = "text-embedding-3-large"

    # -------------------------
    # Load PDF
    # -------------------------

    docs = load_document(FILE_PATH)

    # -------------------------
    # Chunk documents
    # -------------------------

    chunks = text_chunker(docs)

    # -------------------------
    # Load embedding model
    # -------------------------

    embedding_model = load_embedding_model(
        EMBEDDING_MODEL_NAME
    )

    # -------------------------
    # Create FAISS vector store
    # -------------------------

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    # Save vector store
    vector_store.save_local("arad_docs")

    # -------------------------
    # Load LLM
    # -------------------------

    llm_model = load_model()

    # -------------------------
    # Gradio
    # -------------------------

    demo = gr.ChatInterface(
        fn=lambda message, history: chat(
            message,
            history,
            vector_store,
            llm_model
        ),
        title="AradBook AI Assistant",
        description="Ask questions about the provided document."
    )

    demo.launch()


if __name__ == "__main__":
    main()
