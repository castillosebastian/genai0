# coding: utf-8
# -------------------------------------------------------------------------
# Globant
# Advance_RAG
# Backend
# reference: https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_vector_search.py
# --------------------------------------------------------------------------

"""
FILE: retriever.py
DESCRIPTION:
    This backend module load and retrieve data 
    from an Azure Search index.
USAGE:
    python retriever.py
    Set the environment variables with your own values before running the sample:
    1) AZURE_SEARCH_SERVICE_ENDPOINT - the endpoint of your Azure Cognitive Search service
    2) AZURE_SEARCH_INDEX_NAME - the name of your search index (e.g. "hotels-sample-index")
    3) AZURE_SEARCH_API_KEY - your search API key
"""

import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    VectorSearchAlgorithmKind,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters
)
from dotenv import load_dotenv
load_dotenv()

service_endpoint = os.environ["SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
key = os.environ["SEARCH_SERVICE_API_KEY"]


def get_embeddings(text: str):
    # There are a few ways to get embeddings. This is just one example.
    import openai

    open_ai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    open_ai_key = os.getenv("OPENAI_API_KEY")

    client = openai.AzureOpenAI(
        azure_endpoint=open_ai_endpoint,
        api_key=open_ai_key,
        api_version= os.getenv('OPENAI_API_VERSION'),
    )
    embedding = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return embedding.data[0].embedding


def create_index(name: str):
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration,
    )

    fields = [
        # Warning! too many 'filterable's field?
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        SearchableField(name="doc_year", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
        SearchableField(name="doc_type", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="source", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="company_name", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="doc_quarter", type=SearchFieldDataType.String),
        SearchableField(name="page", type=SearchFieldDataType.String),
        SearchableField(name="start_index", type=SearchFieldDataType.String),
        SearchableField(name="page_content", type=SearchFieldDataType.String), # not searchable?
        # SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        #             searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
        SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    ]

    # Configure the vector search configuration  
    # https://learn.microsoft.com/en-us/python/api/azure-search-documents/azure.search.documents.indexes.models.hnswvectorsearchalgorithmconfiguration?view=azure-python-preview
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE
                )
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name="myExhaustiveKnn",
                kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                parameters=ExhaustiveKnnParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE
                )
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm_configuration_name="myExhaustiveKnn",
            )
        ]
    )    
    return SearchIndex(name=name, fields=fields, vector_search=vector_search)

def get_documents():
    # TODO: refactor to add chunked docs /home/sebacastillo/genai0/exp/exp9_dbconfiguration-azure_ai_search/exp_vetors&source-doc&metadata.ipynb
    docs = [
        {
            "hotelId": "1",
            "hotelName": "Fancy Stay",
            "description": "Best hotel in town if you like luxury hotels.",
            "descriptionVector": get_embeddings("Best hotel in town if you like luxury hotels."),
            "category": "Luxury",
        },
        {
            "hotelId": "2",
            "hotelName": "Roach Motel",
            "description": "Cheapest hotel in town. Infact, a motel.",
            "descriptionVector": get_embeddings("Cheapest hotel in town. Infact, a motel."),
            "category": "Budget",
        },       
    ]
    return docs

def single_vector_search(query = None, k_nearest_neighbors=5, top=5):
    # [START single_vector_search]
    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(vector=get_embeddings(query), k_nearest_neighbors=k_nearest_neighbors, fields="contentVector")

    results = search_client.search(
        vector_queries=[vector_query],
        select=["id", "company_name", "source", "doc_type", "page_content"],
        top=top
    )

    for result in results:
        print(result)
    # [END single_vector_search]


def single_vector_search_with_filter(query = None, k_nearest_neighbors=5, top=5, str_to_filter=None):
    # [START single_vector_search_with_filter]    

    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(vector=get_embeddings(query), k_nearest_neighbors=k_nearest_neighbors, fields="contentVector")

    filter_expression = f'category eq {str_to_filter}'

    results = search_client.search(
        search_text="",
        vector_queries=[vector_query],        
        filter=filter_expression,
        select=["id", "company_name", "source", "doc_type", "page_content"],
        top=top
    )

    for result in results:
        print(result)
    # [END single_vector_search_with_filter]


def simple_hybrid_search(query = None, k_nearest_neighbors=5, top=5):
    # [START simple_hybrid_search]
    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(vector=get_embeddings(query), k_nearest_neighbors=k_nearest_neighbors, fields="contentVector")

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["id", "company_name", "source", "doc_type", "page_content"],
        top=top
    )
    print(results.get_answers())
    for result in results:
        print(result)
    # [END simple_hybrid_search]


if __name__ == "__main__":
    credential = AzureKeyCredential(key)
    index_client = SearchIndexClient(service_endpoint, credential)
    #index = create_index('prueba')
    #index_client.create_index(index)
    #client = SearchClient(service_endpoint, index_name, credential)
    #docs = get_documents()
    #client.upload_documents(documents=docs)

    query = "What is the Revenue of Microsoft in 2023"
    k_nearest_neighbors=5
    top=5

    single_vector_search(query=query, k_nearest_neighbors=k_nearest_neighbors, top=top)
    #single_vector_search_with_filter()
    #simple_hybrid_search()