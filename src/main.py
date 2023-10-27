import time
import hydra
import logging
from pathlib import Path
from core.chains import get_chain, get_llm, retrieval_predict
from core.knowledge_base_creator import create_knowledge_base
from core.parsers import create_qa_result_iteration
from core.utils import (
    load_env,
    create_folder_structure,
    retrieve_knowledge_base_path,
    read_questions,
    save_result
)
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from tqdm import tqdm

configuration_file = "new_paragraph_KB_hemophilia_October_2023.yaml"
load_env()  # loads OPENAI & HUGGINGFACE environment variables keys 

log = logging.getLogger(__name__)
# logging.getLogger('langchain').setLevel(logging.ERROR)

@hydra.main(config_path="conf", config_name=configuration_file, version_base=None)
def main(cfg) -> None:

    folders: dict = cfg.folders
    experiment: dict = cfg.experiment
    predictor: dict = cfg.predictor
    knowledgebase: dict = cfg.knowledgebase

    predictor_params = {
        key: value 
        for elem in predictor.config
        for key, value in elem.items()        
    }

    experiment_folder: Path = create_folder_structure(
        folder_path=folders.data, 
        experiment_name=experiment.name
    )

    knowledge_base_path: Path = retrieve_knowledge_base_path(
        experiment_folder=experiment_folder, 
        knowledgebase_data=knowledgebase.data
    )
    log.info("Knowledge Base Path:")
    log.info(knowledge_base_path)

    vectorstore_folder = create_knowledge_base(
        dataset_path=knowledge_base_path,
        embedding_model=knowledgebase.embedding.name
    )
    log.info("VectorStore Folder:")
    log.info(vectorstore_folder)
    vectordb = Chroma(persist_directory=str(vectorstore_folder), embedding_function=OpenAIEmbeddings())

    questions = read_questions(
        experiment_folder = experiment_folder, 
        questions_dataset= experiment.questions
    )

    llm = get_llm(model_name=predictor.name, model_id=predictor.id, predictor_params=predictor_params)

    for question in tqdm(questions):

        try:
            llm_chain = get_chain(llm=llm)
            normal_response:str = llm_chain.predict(question=question)

            #llm_retrieval_chain = get_retrieval_chain(llm=llm, vectorstore_folder=vectorstore_folder)
            #retrieval_response:dict = llm_retrieval_chain({"query": question})
            retrieval_response:dict = retrieval_predict(
                question=question, 
                vectordb=vectordb
            )

            formatted_response:dict = create_qa_result_iteration(
                retrieval_response=retrieval_response,
                normal_response=normal_response
            )

        except Exception as e:
            
            formatted_response = {
                "question": question,
                "with_retrieval_answer": "",
                "pre_llm_answer": "",
                "without_retrieval_answer": "",
                "context": "",
                "source": "",
                "model_name": "",
                "knowledgebase_data": "",
                "embedding_model": ""
            }
            print(e)

        finally:
            save_result(
                model_name = predictor.name,
                knowledgebase_data = knowledgebase.data,
                embedding_model = knowledgebase.embedding.name,
                experiment_folder = experiment_folder,
                response = formatted_response,
                predictor_params=predictor_params
            )
            time.sleep(20)


if __name__ == "__main__":
    
    main()
