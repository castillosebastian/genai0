# %% [markdown]
# 

# %% [markdown]
# # Dataset FinanceBench

# %%
# !pip install qdrant-client

# %%
import pandas as pd
import os
import requests
from datasets import load_dataset
from datasets import DatasetDict
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from dotenv import load_dotenv
load_dotenv()

# Load OpenAI access
import sys
sys.path.append(os.path.abspath('../../src'))
from azure_openai_conn import OpenAIembeddings

# %%
# Turn huggingface dataset to pd
# images = fashion["image"]
# data = fashion.remove_columns("image")
# product_df = data.to_pandas()
# product_data = product_df.reset_index(drop=True).to_dict(orient="index")

if os.path.isfile('../../data/financebench_sample_150.csv'):
    df = pd.read_csv('../../data/financebench_sample_150.csv')
else:    
    ds = load_dataset("PatronusAI/financebench")
    df = pd.DataFrame(ds)
    all_dicts = []
    for index, row in df.iterrows():    
        dictionary = row['train']    
        all_dicts.append(dictionary)
    df = pd.DataFrame(all_dicts)

# %%

destination_folder = '../../data/financebench'

if not os.path.exists(destination_folder):

    os.makedirs(destination_folder)

    for index, row in df.iterrows():
        url = row['doc_link']
        doc_name = row['doc_name']
        doc_name_with_extension = doc_name + '.pdf'        
        file_path = os.path.join(destination_folder, doc_name_with_extension)
        response = requests.get(url)
        if response.status_code == 200:            
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {doc_name_with_extension}")
        else:
            print(f"Failed to download: {doc_name_with_extension} ({url})")


# %%
def load_chunk_persist_pdf(destination_folder) -> Qdrant:
    pdf_folder_path = destination_folder
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    # todo: smarter spliter
    # https://github.com/Azure-Samples/azure-search-openai-demo/blob/main/scripts/prepdocslib/textsplitter.py
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, add_start_index=True # https://github.com/langchain-ai/langchain/blob/master/templates/rag-redis/ingest.py
    )
    
    chunked_documents = text_splitter.split_documents(documents)
    embeddings = OpenAIembeddings()

    qdrant = Qdrant.from_documents(
        chunked_documents,
        embeddings,
        path=destination_folder,
        collection_name="financebench")   

    return qdrant

# %%
qdrant = load_chunk_persist_pdf(destination_folder)


