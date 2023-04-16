import os
import hydra
import logging
from time import time
from pathlib import Path
from core.knowledge_base_creator import create_knowledge_base
from core.utils import (
    load_env,
    create_folder_structure,
    retrieve_dataset_path
)

load_env()

log = logging.getLogger(__name__)

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

    vectorstore_path = create_knowledge_base(
        model_name=model.name,
        dataset_path=dataset_path
    )
    log.info("VectorStore Path:")
    log.info(vectorstore_path)


if __name__ == "__main__":
    main()
