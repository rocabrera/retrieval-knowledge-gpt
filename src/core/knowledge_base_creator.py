import time
from pathlib import Path
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
from tqdm import tqdm



def create_knowledge_base(embedding_model: str, dataset_path: Path) -> Path:
    """
    Create only once a knowledge based on chroma embedding model
    for a given dataset.

    TODO: Later we should be able to change the embedding_model.

    Args:
        embedding_model (str): Name of the model used to create the knowledge base.
        dataset_path (Path): Dataset to be vectorized

    Returns:
        Path: Formated Vectorstore path.
    """

    filename: str = embedding_model + "_" + dataset_path.stem
    experiment_folder: Path = dataset_path.parent.parent
    knowledge_base_folder = experiment_folder / "knowledge_bases"
    vectorstore_folder: Path = (knowledge_base_folder / filename)
    
    if vectorstore_folder.is_dir():
        print("Vectorstore already exists")
        return vectorstore_folder

   # Load Data
    # loader = DirectoryLoader(str(dataset_path), glob="**/*.txt", loader_cls=TextLoader)
    # documents = loader.load()

    df = pd.read_csv(
        str(dataset_path), 
        sep="\t", 
        header=None
    )

    records = df.to_dict(orient="records")
    documents = [
        Document(
            page_content=record[0], 
            metadata={"source": record[1]}) 
        for record in records
    ]
    # TODO Would be better to split the abstracts ?
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # texts = text_splitter.split_documents(documents)

    # Load Data to vectorstore
    embeddings = OpenAIEmbeddings()
    # Trocar Chroma por FAISS
    nbatches = int(len(documents)/100)
    batches_documents = [documents[i*nbatches: nbatches*(i+1)] for i in range(nbatches)]
    for batches_document in tqdm(batches_documents):
        _ = Chroma.from_documents(batches_document, embeddings, persist_directory = str(vectorstore_folder))
        time.sleep(10)
    # DO I NEED THIS?
    # vectordb.persist()
    
    print(f"Vectorstore create on: {vectorstore_folder}")
    return vectorstore_folder
