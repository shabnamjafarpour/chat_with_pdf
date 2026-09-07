import os

import gradio as gr
from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
import arabic_reshaper
from bidi.algorithm import get_display
load_dotenv()

# -----------------------------------------------------
#----------------- Load LLM----------------------------
# -----------------------------------------------------

def load_llm_model(model_name : str = "openai/gpt-4o-mini"):
    model = ChatOpenRouter(
    model= model_name,
    max_tokens=300,
     )
    return model

# -----------------------------------------------------
# --------------Load DOCUMENT--------------------------
# -----------------------------------------------------

def load_document(file_path_: str):
    loader = PyPDFLoader(file_path_)
    documents = loader.load()
    return documents


# -----------------------------------------------------
#---------- Create summary for each page---------------
# -----------------------------------------------------
def summarize_pages(documents_, llm_model_):

    page_summaries = []

    for i, document in enumerate(documents_):

        prompt = f"""
        Summarize the following document page.

        Write the summary in the same language as the original document.

        Focus on:
        - Main topics
        - Important concepts
        - Key entities
        - Important facts
        - Technical terms

        Do not translate the content.
        Do not add information that is not present in the page.

        Page content:
        {document.page_content}

        Summary:
        """

        response = llm_model_.invoke(prompt)

        page_summaries.append({
            "page": i + 1,
            "summary": response.content
        })

    return page_summaries
    
    
# -----------------------------------------------------
# -------------Retrieve relevant pages----------------
# -----------------------------------------------------

def get_retrived_page(
    message_,
    page_summaries_,
    llm_model_,
):
    retrived_Page_num = []
    
    summaries = "\n\n".join(
        f"PAGE {item['page']}:\n{item['summary']} "
        for item in page_summaries_
    )
    
    
    retrieval_prompt = f"""
    You are a document retrieval system.

    You are given summaries of the pages of a document.

    Select the page numbers that are most relevant
    to answering the user's question.

    Return ONLY the page numbers separated by commas.

    For example:
    3, 7, 8

    If no page is relevant, return:
    NONE

    Page summaries:

    {summaries}

    User question:
    {message_}

    Relevant pages:
    """
    
    response = llm_model_.invoke(retrieval_prompt)
    
    return response.content.strip()


    #-----------------------------------------------------
    #--------------------chat----------------------------
    #-----------------------------------------------------
def chat(
    message_,
    history_,
    documents_,
    page_summaries_,
    llm_model_
):
  
    #---------------------------------------------------------
    #--------------Retrive relevent page numbers------------
    #----------------------------------------------------------
  
    relevent_page_numbers = get_retrived_page(
                            message_,
                            page_summaries_,
                            llm_model_)

    #-------------------------------------------------------
    #------------Check if no relevant page found-----------
    #--------------------------------------------------------
    if relevent_page_numbers == "NONE":
         return "I don't know based on the provided document."
    #--------------------------------------------------------
    #-------------Convert page numbers to integers-----------
    #--------------------------------------------------------
    try:
        
        page_numbers = [
        int(page.strip())
        for page in relevent_page_numbers.split(",")
        ]

    except ValueError:
        "I couldn't identify the relevant pages." 
    
    #----------------------------------------------
    #-------------build final context-------------
    #----------------------------------------------

    
    context = "\n\n".join(
        f"PAGE {page_number}:\n"
        f"{documents_[page_number - 1].page_content}"
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
        
    #--------------------------------------------    
    #------------Generate final answer-----------
    #----------------------------------------------
    
    
    answer_response = llm_model_.invoke(
            answer_prompt
    )

    return answer_response.content

    #----------------------------------------------
    #--------------Gradio Interface----------------
    #----------------------------------------------

def create_chat_interface(
    documents_ ,
    page_summarise,
    llm_model_,
):
    
    
    demo = gr.ChatInterface(
        
        fn = lambda message , history :chat(
            message,
            history,
            documents_,
            page_summarise,
            llm_model_
        ),
        
        title="AradBook AI Assistant",
        
        description="Ask questions about the provided document."  
    )
    
    demo.launch()



if __name__=="__main__":
    FILE_PATH = r"D:\AI_REPOSITORY_CACHE_MI\projects\AradBook\docs\external_data_mini.pdf"
    model_llm = load_llm_model()
    documents = load_document(FILE_PATH)
    page_summaries= summarize_pages(documents,model_llm)
    create_chat_interface(documents,page_summaries,model_llm) 


    
