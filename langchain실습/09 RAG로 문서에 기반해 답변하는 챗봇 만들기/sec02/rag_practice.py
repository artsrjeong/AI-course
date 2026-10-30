from langchain_community.document_loaders import PyPDFLoader

# PDF 파일을 읽어서 텍스트 데이터를 추출합니다.
loader = PyPDFLoader('../data/OneNYC_2050_Strategic_Plan.pdf')
data_nyc = loader.load()
#print(data_nyc)

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 텍스트 데이터를 1000자 단위로 나눕니다. overlap은 100자로 설정합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(data_nyc)

for i, split in enumerate(all_splits):
    print(f"Split {i+1}:------------------------------------\n")
    print(split)
    if i>3:
        break

print(type(all_splits[0]))

loader_seoul = PyPDFLoader('../data/2040_seoul_plan.pdf')
data_seoul = loader_seoul.load()
seoul_splits = text_splitter.split_documents(data_seoul)
for i, split in enumerate(seoul_splits):
    print(f"Split {i+1}:------------------------------------")
    print(split)
    if i>3:
        break

print(seoul_splits[50].page_content)
print('----------------------')
print(seoul_splits[51].page_content)

for i in range(len(seoul_splits) - 1):
    seoul_splits[i].page_content += "\n"+ seoul_splits[i + 1].page_content[:100]

print(seoul_splits[50].page_content)
print('----------------------')
print(seoul_splits[51].page_content)


print(len(all_splits))
all_splits.extend(seoul_splits)
print(len(all_splits))


from langchain_huggingface import HuggingFaceEmbeddings

# 다국어 성능이 뛰어난 HuggingFace 무료 모델 로드
embedding = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

v = embedding.embed_query("뉴욕의 온실가스 저감 정책은 뭐야?")
print(v[:5])  # 벡터 일부 출력
print(len(v)) # 차원 수 출력 (1024차원)

from langchain_chroma import Chroma
import os
from pathlib import Path

persist_directory = Path(__file__).resolve().parent.parent / "chroma_store"
database_file = persist_directory / "chroma.sqlite3"
persist_directory.mkdir(parents=True, exist_ok=True)

# 저장된 크로마 DB 파일이 없다면 새로 만들기
if not os.path.isfile(database_file):
    print("Creating new Chroma store")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

else:
    print("Loading existing Chroma store")
    vectorstore = Chroma(		
        persist_directory=persist_directory, 
        embedding_function=embedding
    )

# %%
retriever = vectorstore.as_retriever(k=3)
docs = retriever.invoke("서울시의 환경 정책에 대해 궁금해")

for d in docs:
    print(d)
    print('------')
