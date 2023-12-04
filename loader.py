import json
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders import TextLoader, PyPDFLoader


# os.environ["OPENAI_API_KEY"] = 'sk-'
loader = PyPDFLoader(r"C:\Users\Vrdella\Downloads\payslip.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embeddings)
db.save_local('index_store')
# query = "what is the total amount"
# docs = db.similarity_search(query)
#
# print(docs[0].page_content)
#
#
# pkl = db.serialize_to_bytes()
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#
# db1 = FAISS.from_texts(["resume_text"], embeddings)
#
# print(db1)
#
# vector_embedding = db.docstore._dict
# print(vector_embedding)
#
# texts = text_splitter.create_documents(vector_embedding)
# vector_index = FAISS.from_documents(texts, OpenAIEmbeddings())
# vector_index.save_local('index_store')

# json_object = json.dumps(vector_embedding, indent=4)

# Writing to sample.json
# with open("sample.json", "w") as outfile:
#     outfile.write(vector_embedding)


