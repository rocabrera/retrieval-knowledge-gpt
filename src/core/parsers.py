import pandas as pd
from pathlib import Path

def create_qa_result_iteration(retrieval_response: dict, normal_response:str):
    document = retrieval_response["source_documents"][0]
    page_content = (document.page_content)
    source = document.metadata["source"]
    retrieval_response["context"] = page_content
    retrieval_response["source"] = source
    _ = retrieval_response.pop("source_documents")
    retrieval_response["question"] = retrieval_response.pop("query")
    retrieval_response["with_retrieval_answer"] = retrieval_response.pop("result")

    retrieval_response["without_retrieval_answer"] = normal_response

    return retrieval_response

