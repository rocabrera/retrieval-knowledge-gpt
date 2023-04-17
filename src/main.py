import os
import pickle
import hydra
import logging
from time import time
from pathlib import Path
from core.chains import get_retrieval_chain, get_chain
from core.knowledge_base_creator import create_knowledge_base
from core.parsers import create_qa_result_iteration, save_result
from core.utils import (
    load_env,
    create_folder_structure,
    retrieve_dataset_path
)

load_env()

log = logging.getLogger(__name__)
logging.getLogger('langchain').setLevel(logging.ERROR)

@hydra.main(config_path="conf", config_name="flan-t5-xl_config.yaml", version_base=None)
def main(cfg) -> None:

    folders: dict = cfg.folders
    experiment: dict = cfg.experiment
    model: dict = cfg.model

    experiment_folder: Path = create_folder_structure(
        folder_path=folders.data, 
        experiment_name=experiment.name
    )

    dataset_path: Path = retrieve_dataset_path(
        experiment_folder=experiment_folder, 
        dataset_name=experiment.dataset
    )
    log.info("Dataset Path:")
    log.info(dataset_path)

    vectorstore_folder = create_knowledge_base(
        model_name=model.name,
        dataset_path=dataset_path
    )
    log.info("VectorStore Folder:")
    log.info(vectorstore_folder)
    
    # question = "What it is meditation"
    question = input("Question:")

    llm_retrieval_chain = get_retrieval_chain(vectorstore_folder=vectorstore_folder)
    retrieval_response:dict = llm_retrieval_chain({"query": question})

    llm_chain = get_chain()
    normal_response:str = llm_chain.predict(question=question, context="")

    formatted_response:dict = create_qa_result_iteration(
        retrieval_response=retrieval_response,
        normal_response=normal_response
    )

    print(formatted_response)

    save_result(
        model_name=model.name,
        experiment_folder=experiment_folder,
        response=formatted_response
    )

    

if __name__ == "__main__":
    main()
