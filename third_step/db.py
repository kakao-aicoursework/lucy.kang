import chromadb
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

CHROMA_PERSIST_DIR = "./chroma"

db_client = chromadb.PersistentClient()
collection = db_client.get_or_create_collection(
    name="kakao",
    metadata={"hnsw:space": "cosine"}
)

def getDB(collection_name):
    makeDB()
    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(),
        collection_name=collection_name
    )
    return db

def makeDB():
    # Vector DB로 데이터를 저장
    data_types = ["channel", "social", "sync"]

    for data_type in data_types:
        file_path = f"./file/{data_type}.txt"
        documents = TextLoader(file_path).load()

        ids = pre_process_ids(data_type, documents)

        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=160)  # 데이터 청킹할때 사용
        docs = text_splitter.split_documents(documents)
        # ids = [f"{data_type}-{i}" for i in range(len(docs))]

        Chroma.from_documents(
            docs,
            OpenAIEmbeddings(),  # openaid에서 사용하는 임베딩 방식으로
            collection_name=data_type,
            persist_directory=CHROMA_PERSIST_DIR,
            ids=ids,
        )


def pre_process_ids(data_type, documents):
    print(documents)
    ids = []

    for doc in documents:
        for chunk in doc.page_content.split("\n#")[2:]:
            title = chunk.split("\n")[0].replace(" ", "-").strip()
            _id = f"{data_type}-{title}"
            _doc = chunk.strip()
            doc.page_content = _doc
            ids.append(_id)
    return ids
