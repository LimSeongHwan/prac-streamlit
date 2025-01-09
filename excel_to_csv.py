import pandas as pd
import numpy as np

# 파일 불러오기
df = pd.read_excel("./data/excel_test3.xlsx")

# 첫 번째 row의 모든 값이 'Unnamed'로 시작하는지 확인
if df.columns.str.startswith('Unnamed').all():
    # 첫 번째 행을 삭제하고, 그 다음 행을 새로운 헤더로 설정
    df = pd.read_excel("./data/excel_test.xlsx")
    # 모든 값이 NaN인 행 삭제
    df = df.dropna(axis=0, how='all')
    # 모든 값이 NaN인 열 삭제
    df = df.dropna(axis=1, how='all')


# 엑셀 데이터 중 빈 값은 NaN으로 표시되는데 임베딩 중 에러로 인해 이 값을 string NaN으로 변경
df = df.replace(np.NaN, '')
df = df.replace('-', '')
df = df.replace('nan', '')
df[['협약사', '기술구분_SI(인력)', '기술구분_솔루션', '기술구분_솔루션명']] = df[['협약사', '기술구분_SI(인력)', '기술구분_솔루션', '기술구분_솔루션명']].replace('O', '1')
df[['협약사', '기술구분_SI(인력)', '기술구분_솔루션', '기술구분_솔루션명']] = df[['협약사', '기술구분_SI(인력)', '기술구분_솔루션', '기술구분_솔루션명']].replace('X', '0')


# print(df)
# 엑셀 특성 상 셀 병합으로 인해 column명이 2개의 행을 차지하는 경우가 있음, 이 경우 데이터를 읽었을 때, Unnamed로 표시됨, Unnamed 필터링
# 엑셀 특성 상 컬럼 이름을 분류하는 상위 컬럼이 존재 할 수 있음 Ex) 분류 -> 대 중 소, 이런 컬럼들을 공백을 사이에 두고 합쳐준다. Ex) 분류 -> 대 중 소 --> 분류 대, 분류 중, 분류 소
# df.columns = ['_'.join(col).strip() if not col[1].startswith('Unnamed') else col[0] for col in df.columns.values]
#
# for column in df.columns:
#     df[column] = df[column].replace(',', ' ')
#
df.to_csv(path_or_buf='./data/excel_test3.csv', sep = ',', header=True, index=False, mode='w', encoding='UTF-8')