from pathlib import Path
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings


def create_knowledge_base(model_name: str, dataset_path: Path):

    filename: str = model_name + "_" + dataset_path.stem + ".pickle"
    experiment_folder: Path = dataset_path.parent.parent
    knowledge_base_folder = experiment_folder / "knowledge_bases"
    vectorstore_path: Path = (knowledge_base_folder / filename)
    if vectorstore_path.is_file():
        print("Vectorstore already exists")
        return vectorstore_path

   # Load Data
    loader = UnstructuredFileLoader(dataset_path)
    raw_documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=150, chunk_overlap=0)
    texts = text_splitter.split_documents(raw_documents)

    # Load Data to vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Save vectorstore
    with open(vectorstore_path, "wb") as f:
        pickle.dump(vectorstore, f)

    print(f"Vectorstore create on: {vectorstore_path}")
    return vectorstore_path
