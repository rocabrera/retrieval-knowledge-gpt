import os
from typing import List
from pathlib import Path
from langchain import PromptTemplate
import pandas as pd


def create_pre_prompt_template(question) -> PromptTemplate:

    pre_prompt = f"""
    From now on, act as a hemophilia expert, with experience in clinical practice and molecular biology.
    Answer the question below concisely and objectively. Your answer should be a single paragraph, with between 150-300 words.
    The audience are hemophilia researchers very familiar with the topic.
    
    {question}
    """
    
    return pre_prompt

def create_main_prompt_template(question, context) -> PromptTemplate:

    main_prompt = f"""
    From now on, act as a hemophilia expert, with experience in clinical practice and molecular biology.
    Answer the question below using the context given between triple # as the basis to compose your answer. Your answer should have 500 words or more.
    
    The tone should be professional, serious and friendly. The audience are medical workers and molecular biology researchers somewhat familiar with hemophilia. However, explain clearly any terms not known to people outside hemophilia research.
    
    ###
    {context}
    ###

    {question}
    """
    
    return main_prompt

def create_main_prompt_without_context_template() -> PromptTemplate:

    main_prompt = """
    From now on, act as a hemophilia expert, with experience in clinical practice and molecular biology. 
    Your answer should have 500 words or more.
    
    The tone should be professional, serious and friendly. The audience are medical workers and molecular biology researchers somewhat familiar with hemophilia. However, explain clearly any terms not known to people outside hemophilia research.
    
    {question}
    """
    
    return PromptTemplate(
        template=main_prompt, input_variables=["question"]
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
    response: dict,
    predictor_params:dict
    ):

    predictor_params_text = "_".join([f"{key}_{value}" for key,value in predictor_params.items()])
    result_path = experiment_folder / "results" / "_".join([model_name, str(Path(knowledgebase_data).stem), embedding_model, predictor_params_text, "results.csv"])
    

    # TODO essa parte ta estranha... deveria estar aqui? 
    response["model_name"] = model_name
    response["knowledgebase_data"] = knowledgebase_data
    response["embedding_model"] = embedding_model
    
    data = pd.DataFrame([response])
    if result_path.is_file():
        data.to_csv(result_path, mode="a", header=False, index=False)
    else:
        data.to_csv(result_path, index=False)
