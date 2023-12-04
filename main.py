# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
#
# # List of PDF file paths
# pdf_files = [
#     r"C:\Users\Vrdella\Downloads\Booklet Housing Programs Services in AD.pdf",
#     r"C:\Users\Vrdella\Downloads\output_compressed.pdf",
#     r"C:\Users\Vrdella\Downloads\payslip.pdf",
# ]
#
# # Searching for a keyword
# query = 'ertyui'
#
# for file in pdf_files:
#     pdf_loader = PyPDFLoader(file)
#     documents = pdf_loader.load()
#
#     # Split documents into chunks for efficient indexing
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     docs = text_splitter.split_documents(documents)
#
#     # Use HuggingFace embeddings
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#
#     # Create FAISS index from documents
#     faiss_index = FAISS.from_documents(docs, embeddings)
#     faiss_index.save_local('index_store')
#
#     vector_index = FAISS.load_local("index_store", embeddings)
#     retriever = vector_index.as_retriever(search_type="similarity")
#     retrieved_docs = retriever.get_relevant_documents(query)
#     for doc in retrieved_docs:
#         if query.lower() in doc.page_content.lower():
#             print(f"Document: {doc.metadata['source']}")
#             # print(f"Content: {doc.page_content}")
#
#==========================================================================================================
import os

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# List of PDF file paths
pdf_files = [
    r"C:\Users\Vrdella\Downloads\Booklet Housing Programs Services in AD.pdf",
    r"C:\Users\Vrdella\Downloads\output_compressed.pdf",
    r"C:\Users\Vrdella\Downloads\payslip.pdf",
    r"C:\Users\Vrdella\Downloads\olx.pdf"
]

# Searching for a keyword
query = 'Employee Name'
os.environ["OPENAI_API_KEY"] = 'sk-'

for file in pdf_files:
    pdf_loader = PyPDFLoader( file )
    documents = pdf_loader.load()

    # Split documents into chunks for efficient indexing
    text_splitter = CharacterTextSplitter( chunk_size=1000, chunk_overlap=0 )
    docs = text_splitter.split_documents( documents )

    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings( model_name="all-MiniLM-L6-v2" )

    # Create FAISS index from documents
    faiss_index = FAISS.from_documents( docs, embeddings )
    faiss_index.save_local( 'index_store' )

    vector_index = FAISS.load_local( "index_store", embeddings )
    retriever = vector_index.as_retriever(search_type="similarity")
    retrieved_docs = retriever.get_relevant_documents( query )

    for doc in retrieved_docs:
        if query.lower() in doc.page_content.lower():
            print( f"Document: {doc.metadata['source']}" )
            print( f"Page No: {doc.metadata['page']}" )  # Print the page number
            # print(f"Content: {doc.page_content}")
            qa_interface = RetrievalQA.from_chain_type(
                llm=ChatOpenAI( model_name='gpt-3.5-turbo-16k' ),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            response = qa_interface( query )
            print( response["result"] )