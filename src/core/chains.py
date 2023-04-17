from pathlib import Path
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from core.utils import create_prompt_template

def get_llm(model:str):

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"temperature":1e-10}
    )
    
    return llm


def get_retrieval_chain(vectorstore_folder:Path):

    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=str(vectorstore_folder), embedding_function=embedding)
    
    chain_type_kwargs = {"prompt": create_prompt_template()}
    llm_retrieval_chain = RetrievalQA.from_chain_type(
        llm=get_llm("dummy"), 
        chain_type="stuff", 
        retriever=vectordb.as_retriever(), 
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    return llm_retrieval_chain


def get_chain():


    llm_chain = LLMChain(
        prompt=create_prompt_template(),
        llm=get_llm("dummy")
    )

    return llm_chain

