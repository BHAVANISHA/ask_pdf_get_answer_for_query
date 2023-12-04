from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, PaiEasChatEndpoint, ChatFireworks, GigaChat, ChatLiteLLM, ChatVertexAI, \
    ChatOllama, FakeListChatModel, HumanInputChatModel
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
import os

# os.environ["OPENAI_API_KEY"] = 'sk-'
query = 'Employee Name '
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_index = FAISS.load_local("index_store", embeddings)
retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 6})  #, search_kwargs={"k": 6})
qa_interface = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name='gpt-3.5-turbo-16k'),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)
response = qa_interface(query)
print(response["result"])
source_list = []
qa_interface = vector_index.similarity_search_with_relevance_scores(query, k=100)
for document in qa_interface:
    if document[1] > 0.65:
        source_list.append(document[0].metadata['source'])
source_list = list(set(source_list))
print('success')
source_list1 = []
for key, index in vector_index.docstore._dict.items():
    if query in index.page_content:
        source_list1.append(index.metadata['source'])
        print('success')

source_list1 = list(set(source_list1))
print(source_list1)
