from langchain_community.document_loaders import DataFrameLoader
import numpy as np
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import pandas as pd

# 파일 불러오기
# df = pd.read_excel("./data/excel_test.xlsx")
df = pd.read_csv("./data/excel_test3.csv")

# 첫 번째 row의 모든 값이 'Unnamed'로 시작하는지 확인
# if df.columns.str.startswith('Unnamed').all():
    # 첫 번째 행을 삭제하고, 그 다음 행을 새로운 헤더로 설정
    # df = pd.read_excel("./data/excel_test.xlsx", header=[1,2])
    # 모든 값이 NaN인 행 삭제
#     df = df.dropna(axis=0, how='all')
#     # 모든 값이 NaN인 열 삭제
#     df = df.dropna(axis=1, how='all')
#
# 엑셀 데이터 중 빈 값은 NaN으로 표시되는데 임베딩 중 에러로 인해 이 값을 string NaN으로 변경
df = df.replace(np.NaN, '')

# # 엑셀 특성 상 셀 병합으로 인해 column명이 2개의 행을 차지하는 경우가 있음, 이 경우 데이터를 읽었을 때, Unnamed로 표시됨, Unnamed 필터링
# # 엑셀 특성 상 컬럼 이름을 분류하는 상위 컬럼이 존재 할 수 있음 Ex) 분류 -> 대 중 소, 이런 컬럼들을 공백을 사이에 두고 합쳐준다. Ex) 분류 -> 대 중 소 --> 분류 대, 분류 중, 분류 소
# df.columns = ['_'.join(col).strip() if not col[1].startswith('Unnamed') else col[0] for col in df.columns.values]

# 데이터프레임으로 변환
loader = DataFrameLoader(df, page_content_column='순번')

# 데이터 프레임을 리스트로 변환
document_list = loader.load()

# # 환경변수를 불러옴
load_dotenv()

# # OpenAI에서 제공하는 Embedding Model을 활용해서 `chunk`를 vector화
embedding = UpstageEmbeddings(model="solar-embedding-1-large")
# # embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# # 임베딩
index_name = 'excel-test-idx'
database = PineconeVectorStore.from_documents(document_list, embedding, index_name=index_name)