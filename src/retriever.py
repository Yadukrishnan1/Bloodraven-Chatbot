import pickle
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "data/faiss_index"
METADATA_PATH = "data/metadata.pkl"

cached_retriever = None

def setup_retriever():
    global cached_retriever
    if cached_retriever is None:
        db = FAISS.load_local(DB_PATH, HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))
        with open(METADATA_PATH, "rb") as f:
            chunked_docs = pickle.load(f)
        cached_retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 4})
    return cached_retriever
