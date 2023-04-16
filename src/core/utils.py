import os
from pathlib import Path

def load_env():

    with open(".env", "r") as f:
        var_envs = f.readlines()
        for var in var_envs:
            name, value = var.strip("\n").split("=")
            os.environ[name] = value.strip('"')


def create_folder_structure(folder_path:str, experiment_name: str):
    """
    Create the folder repository structure for a given experiment.

    Args:
        folder_path (str): Data root folder of project.
        experiment_name (str): Experiment name.

    Returns:
        Path: Experiment folder.
    """

    experiment_folder = Path(f"{folder_path}/{experiment_name}")

    dataset_folder = experiment_folder / "datasets"
    knowledge_bases_folder = experiment_folder / "knowledge_bases"
    questions_folder = experiment_folder / "questions"

    experiment_folder.mkdir(parents=True, exist_ok=True)
    dataset_folder.mkdir(parents=True, exist_ok=True)
    knowledge_bases_folder.mkdir(parents=True, exist_ok=True)
    questions_folder.mkdir(parents=True, exist_ok=True)

    (dataset_folder / ".gitkeep").touch(exist_ok = True)
    (knowledge_bases_folder / ".gitkeep").touch(exist_ok = True)
    (questions_folder / ".gitkeep").touch(exist_ok = True)

    return experiment_folder

def retrieve_dataset_path(experiment_folder: str, dataset_name: str):
    """
    If dataset of given experiment exists retrieve path formated.

    Args:
        experiment_folder (str): Data experiment folder.
        dataset_name (str): Dataset name

    Returns:
        Path: Formated dataset path.

    Raises:
        Exception: If dataset is not set in the correct folder.
    """
    
    dataset_path: Path = Path(f"{experiment_folder}/datasets/{dataset_name}")
    if dataset_path.is_file():
        return dataset_path

    raise Exception(f"Dataset not found on path: {experiment_folder}/datasets/{dataset_name}")