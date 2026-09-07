import os

import gradio as gr
from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()



def load_llm_model(model_name : str = "openai/gpt-4o-mini"):
    model = ChatOpenRouter(
    model= model_name,
    max_tokens=300,
     )
    return model

def load_document(file_path_: str):
    loader = PyPDFLoader(file_path_)
    documents = loader.load()
    return documents

    
    
    
def chat(message_, history_, documents_, llm_model):
    """
        Retrieve relevant pages and answer the user's question.
    """
    #-----------------------------------------------------
    #-----------------Build pages for retrieval-----------
    #-----------------------------------------------------
    pages = "\n\n".join(
        f"PAGE {i + 1}:\n{doc.page_content[:1500]}"
        for i, doc in enumerate(documents_)
    )

    retrieval_prompt  = f"""
        You are a document retrieval system.

        You are given a document divided into pages.

        Select the page numbers that are most relevant to answering
        the user's question.

        Return ONLY the page numbers separated by commas.

        For example:
        3, 7, 8

        If no page is relevant, return:
        NONE

        Document:

        {pages}

        User question:
        {message_}

        Relevant pages:
        """
    #---------------------------------------------------------
    #--------------Ask LLM which pages are relevant------------
    #----------------------------------------------------------

    retrival_response = llm_model.invoke(
        retrieval_prompt 
    )
    relevent_page = retrival_response.content.strip()

    #-------------------------------------------------------
    #------------Check if no relevant page exists-----------
    #--------------------------------------------------------
    if relevent_page == "NONE":
         return "I don't know based on the provided document."
    #--------------------------------------------------------
    #-------------Convert page numbers to integers-----------
    #--------------------------------------------------------
    try:
        
        page_numbers = [
        int(page.strip())
        for page in relevent_page.split(",")
        ]

    except ValueError:
        "I couldn't identify the relevant pages."
        
    #---------------------------------------------  
    #-------------Retrieve actual pages-----------
    #----------------------------------------------
    selected_documents = [
        documents_[page_number - 1]
        for page_number in page_numbers
        if 0 < page_number <= len(documents_)
    ]    
    
    #----------------------------------------------
    #-------------build final context-------------
    #----------------------------------------------

    context = "\n\n".join(
        f"selected document is {selected_documents}"
    )
    
    context = "\n\n".join(
    f"PAGE {page_number}:\n{documents_[page_number - 1].page_content}"
    for page_number in page_numbers
    if 0 < page_number <= len(documents_)
    )
    # """
    # يعني بيا از بين صفحات مرتبطي كه در مرحله قبل به دست آروده ايم،‌
    # دونه دونه محتويات كامل صفحات رو وردار همراه شماره آن صفحه
    # و همه اين هارو به هم جوين كن و به عنوان محتواي مرتبط بده به
    # مدل تا پاسخ نهايي را توليد كند :)
    # """
    #----------------------------------------------
    #------------final answer prompt---------------
    #----------------------------------------------
    answer_prompt = f"""
        You are a helpful assistant.

        Answer the user's question based only on the provided context.

        If the answer cannot be found in the context, say:

        "I don't know based on the provided document."

        Context:
        {context}

        Question:
        {message_}

        Answer:
        """
    # 9. Generate final answer
    answer_response = llm_model.invoke(
            answer_prompt)

    return answer_response.content



def create_chat_interface(documents_ ,llm_model_):
    demo = gr.ChatInterface(
        fn = lambda message , history :chat(
            message,
            history,
            documents_,
            llm_model_
        ),
        title="AradBook AI Assistant",
        description="Ask questions about the provided document."  
    )
    demo.launch()


if __name__=="__main__":
    FILE_PATH = r"D:\AI_REPOSITORY_CACHE_MI\projects\AradBook\docs\external_data.pdf"
    model_llm = load_llm_model()
    documents = load_document(FILE_PATH)
    create_chat_interface(documents,model_llm) 
    

    
