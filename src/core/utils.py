import os
from typing import List
from pathlib import Path
from langchain import PromptTemplate
import pandas as pd


def create_prompt_template() -> PromptTemplate:

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer in English:"""
    return PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


def load_env():

    with open(".env", "r") as f:
        var_envs = f.readlines()
        for var in var_envs:
            name, value = var.strip("\n").split("=")
            os.environ[name] = value.strip('"')


def read_questions(experiment_folder:Path, questions_dataset: str) -> List[str]:
    """
    Read a txt file with one question per line .

    Args:
        experiment_folder (Path): Experiment folder.
        questions_dataset (str): Question dataset name.

    Returns:
        list: List of questions.
    """

    questions = experiment_folder / "questions" / questions_dataset
    if questions.is_file():
        with open(questions, "r") as file:
            data = [line.strip() for line in file.readlines()]
        return data

    return Exception(f"Questions dataset not found on path: {experiment_folder}/questions/{questions_dataset}")

def create_folder_structure(folder_path:str, experiment_name: str) -> Path:
    """
    Create the folder repository structure for a given experiment.

    Args:
        folder_path (str): Data root folder of project.
        experiment_name (str): Experiment name.

    Returns:
        Path: Experiment folder.
    """

    experiment_folder = Path(f"{folder_path}/{experiment_name}")
    experiment_folder.mkdir(parents=True, exist_ok=True)

    folders_of_experiment = ["datasets", "knowledge_bases", "results", "questions"]

    for folder in folders_of_experiment:
        pathlib_folder = experiment_folder / folder
        pathlib_folder.mkdir(parents=True, exist_ok=True)
        (pathlib_folder / ".gitkeep").touch(exist_ok = True)

    return experiment_folder

def retrieve_knowledge_base_path(experiment_folder: str, knowledgebase_data: str) -> Path:
    """
    If knowledge base dataset of given experiment exists retrieve path formated.

    Args:
        experiment_folder (str): Experiment folder.
        knowledgebase_data (str): Dataset used to create a knowledgebase

    Returns:
        Path: Formated dataset path.

    Raises:
        Exception: If dataset is not set in the correct folder.
    """
    
    dataset_path: Path = Path(f"{experiment_folder}/datasets/{knowledgebase_data}")
    if dataset_path.is_file() or dataset_path.is_dir():
        return dataset_path

    raise Exception(f"Dataset not found on path: {experiment_folder}/datasets/{knowledgebase_data}")


def save_result(
    model_name: str, 
    knowledgebase_data:str, 
    embedding_model:str,
    experiment_folder:Path, 
    response: dict):

    result_path = experiment_folder / "results" / "_".join([model_name, str(Path(knowledgebase_data).stem), embedding_model, "results.csv"])
    
    # TODO essa parte ta estranha... deveria estar aqui? 
    response["model_name"] = model_name
    response["knowledgebase_data"] = knowledgebase_data
    response["embedding_model"] = embedding_model
    
    data = pd.DataFrame([response])
    if result_path.is_file():
        data.to_csv(result_path, mode="a", header=False, index=False)
    else:
        data.to_csv(result_path, index=False)
