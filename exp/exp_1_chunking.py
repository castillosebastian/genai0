
import os
import openai
import sys
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'


# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("bd/Amazon_2023.pdf"),
    PyPDFLoader("bd/Apple_2023.pdf"),    
    PyPDFLoader("bd/Tesla_2022.pdf"),
    PyPDFLoader("bd/Moderna_2022.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits1500_150 = text_splitter.split_documents(docs)
len(splits1500_150)

# Split
text_splitter_2 = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 200
)
splits2000_200 = text_splitter_2.split_documents(docs)
len(splits2000_200)

# Save DB
vectordb_split_1500_150 = Chroma.from_documents(
    documents=splits1500_150,
    embedding=embedding,
    persist_directory= persist_directory
)
print(vectordb_split_1500_150._collection.count())


vectordb_splits2000_200 = Chroma.from_documents(
    documents=splits2000_200,
    embedding=embedding,
    persist_directory= persist_directory
)
print(vectordb_splits2000_200._collection.count())


# Evaluate Responses




question = "What is the revenue of Apple"
docs = vectordb_split_1500_150.similarity_search(question,k=3)
len(docs)
docs[0]
docs[1]
docs[2]
for doc in docs:
    print(doc.metadata)



vectordb_split_1500_150.persist()
vectordb_splits2000_200.persist()


