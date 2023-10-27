def create_qa_result_iteration(retrieval_response: dict, normal_response:str):
    # source_documents = retrieval_response["source_documents"]
    # page_contents = [document.page_content for document in source_documents]
    # sources = [document.metadata["source"] for document in source_documents]
    # retrieval_response["contexts"] = page_contents
    # retrieval_response["sources"] = sources
    # _ = retrieval_response.pop("source_documents")
    # retrieval_response["question"] = retrieval_response.pop("query")
    # retrieval_response["with_retrieval_answer"] = retrieval_response.pop("result")

    retrieval_response["without_retrieval_answer"] = normal_response

    return retrieval_response

