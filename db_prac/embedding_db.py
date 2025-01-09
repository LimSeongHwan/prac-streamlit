from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)

loader = Docx2txtLoader('./test.docx')
document_list = loader.load_and_split(text_splitter=text_splitter)

# 데이터 프레임을 리스트로 변환
document_list = loader.load()

# # 환경변수를 불러옴
load_dotenv()

# # OpenAI에서 제공하는 Embedding Model을 활용해서 `chunk`를 vector화
embedding = OpenAIEmbeddings(model='text-embedding-3-large')
# # embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# # 임베딩
index_name = 'table-idx'
database = PineconeVectorStore.from_documents(document_list, embedding, index_name=index_name)