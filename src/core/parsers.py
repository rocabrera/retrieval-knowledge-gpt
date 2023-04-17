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


def save_result(model_name: str, experiment_folder:Path, response: dict):
    result_path = experiment_folder / "results" / (model_name + "_questions.csv")

    data = pd.DataFrame([response])
    if result_path.is_file():
        data.to_csv(result_path, mode="a", header=False, index=False)
    else:
        data.to_csv(result_path, index=False)
