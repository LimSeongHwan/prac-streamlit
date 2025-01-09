from langchain_community.document_loaders import DataFrameLoader
import numpy as np
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import pandas as pd
from llama_index import LlamaIndex
from llama_index.excel import ExcelLoader

index = LlamaIndex()


df = pd.read_csv("./data/excel_test3.csv")
df = df.replace(np.NaN, '')

loader = ExcelLoader("./data/excel_test3.xlsx")
document_list = loader.load_data()

print(document_list)
# # 환경변수를 불러옴
# load_dotenv()

# # OpenAI에서 제공하는 Embedding Model을 활용해서 `chunk`를 vector화
# embedding = UpstageEmbeddings(model="solar-embedding-1-large")
# # embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# # 임베딩
index_name = 'excel-test-idx'
database = PineconeVectorStore.from_documents(document_list, embedding, index_name=index_name)