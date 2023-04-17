from pathlib import Path
import pickle
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def create_knowledge_base(model_name: str, dataset_path: Path):

    filename: str = model_name + "_" + dataset_path.stem
    experiment_folder: Path = dataset_path.parent.parent
    knowledge_base_folder = experiment_folder / "knowledge_bases"
    vectorstore_folder: Path = (knowledge_base_folder / filename)
    if vectorstore_folder.is_dir():
        print("Vectorstore already exists")
        return vectorstore_folder

   # Load Data
    loader = UnstructuredFileLoader(str(dataset_path))
    raw_documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(raw_documents)

    # Load Data to vectorstore
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(texts, embeddings, persist_directory = str(vectorstore_folder))
    vectordb.persist()
    
    print(f"Vectorstore create on: {vectorstore_folder}")
    return vectorstore_folder
