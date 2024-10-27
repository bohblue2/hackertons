import faiss
from constant import DEFAULT_EMBEDDING_DIM, DEFAULT_FAISS_FOLDER_PATH, DEFAULT_FAISS_INDEX_NAME, OPENAI_EMBEDDING_DIM, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
file_paths = [
    '2-14-00-004-20240927.PDF'
]

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

loader = PyPDFLoader(
    file_path=file_paths[0], 
    extract_images=False
)
# loader = PyPDFDirectoryLoader('data/')
docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=30
)
split_docs = loader.load_and_split(text_splitter=text_splitter)
print(len(split_docs))

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings

# 임베딩
embeddings = OpenAIEmbeddings()

db = FAISS(
    embedding_function=embeddings,
    index=faiss.IndexFlatL2(DEFAULT_EMBEDDING_DIM),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
db.add_documents([
    *split_docs
], ids=[str(_id) for _id in range(len(split_docs))])
# print(db.docstore._dict)

db.save_local(
    folder_path=DEFAULT_FAISS_FOLDER_PATH,
    index_name=DEFAULT_FAISS_INDEX_NAME
)