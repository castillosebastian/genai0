# install
# ! pip install langchain unstructured[all-docs] pydantic lxml langchainhub
# ! brew install tesseract
# ! brew install poppler

import os
import sys

import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch, Chroma

# Load environment variables
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Instantiations
embedding = OpenAIEmbeddings()
persist_directory = 'docs/chroma/'
sys.path.append('../..')

# model
llm_name = "gpt-3.5-turbo"

# ETL 
def load_db(files_dir, chain_type, k):
    
    # load documents
    documents = []
    for file in os.listdir('bd'):
        if file.endswith('.pdf'):
            pdf_path = files_dir + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = files_dir + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = files_dir + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 






# Table Recognition: 1:   
# https://blog.langchain.dev/benchmarking-rag-on-tables/



# Table Recognition 2:
# Script /home/sebacastillo/genai0/related_works/Cloud_VM/Unstructured_io_+_LlamaIndex_Llama_Pack_for_Complex_PDF_Retrieval_Step_by_Step.ipynb
