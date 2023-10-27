import openai
from pathlib import Path
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import Tuple, List
from core.utils import (
    create_main_prompt_template, 
    create_pre_prompt_template, 
    create_main_prompt_without_context_template
)

# import tiktoken


def get_llm(model_name:str, model_id:str, predictor_params:dict):

    # if model_name == "flan_t5_xl":
    #     return HuggingFaceHub(
    #         repo_id=model_id,
    #         model_kwargs={"temperature":1e-10}
    #     )
    if model_name == "gpt3.5_turbo" or model_name == "text_davinci_003":

        model = OpenAI(model_name=model_id, model_kwargs=predictor_params)
        # encoding = tiktoken.encoding_for_model(model_id)
        return model
    raise Exception(f"Invalid model selected.")

def predict(question):

    llm_answer = get_completion(create_main_prompt_without_context_template(question))

    return llm_answer


def get_retrieval_chain(llm, vectorstore_folder:Path):

    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=str(vectorstore_folder), embedding_function=embedding)
    
    chain_type_kwargs = {"prompt": create_main_prompt_template()}

    llm_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectordb.as_retriever(), 
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    return llm_retrieval_chain


def get_chain(llm):

    llm_chain = LLMChain(
        prompt=create_main_prompt_without_context_template(),
        llm=llm
    )

    return llm_chain

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def retrieval_predict(question, vectordb):

    pre_llm_answer = get_completion(create_pre_prompt_template(question))

    documents: List[Tuple[Document, float]] = vectordb.similarity_search_with_score(pre_llm_answer, search_kwargs={"k": 3})

    context = "\n".join(document[0].page_content for document in documents)
    source = "\n".join(document[0].metadata["source"] for document in documents)

    main_llm_answer:str = get_completion(
        create_main_prompt_template(
            question=question, 
            context=context
        )
    )

    retrieval_response = {
        "question": question,
        "with_retrieval_answer": main_llm_answer,
        "pre_llm_answer": pre_llm_answer,
        "context" : context,
        "source": source
    } 

    return retrieval_response