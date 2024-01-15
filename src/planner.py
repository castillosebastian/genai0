# coding: utf-8
# -------------------------------------------------------------------------
# Advance_RAG
# Backend
# reference: 
#  notebooks example https://github.com/microsoft/semantic-kernel/tree/main/python/notebooks
#  
#  https://learn.microsoft.com/en-us/semantic-kernel/agents/plugins/?tabs=python
#  https://learn.microsoft.com/en-us/semantic-kernel/agents/plugins/?tabs=Csharp

# Azure connectors:
# https://github.com/microsoft/semantic-kernel/blob/main/python/samples/kernel-syntax-examples/azure_cognitive_search_memory.py

# [Doc](https://learn.microsoft.com/es-mx/semantic-kernel/)
# [Repo](https://github.com/microsoft/semantic-kernel)
# [Repo-Python-Examples](https://github.com/microsoft/semantic-kernel/blob/main/python/README.md), 
# [Examples2](https://github.com/microsoft/semantic-kernel/tree/main/python/samples/kernel-syntax-examples)
# --------------------------------------------------------------------------

"""
FILE: planner.py
DESCRIPTION:
    The script map the user 'ask' (input) to given pipeline or skills to produce and output    
USAGE:
    
TODO:
    
"""
import os
from tqdm import tqdm
import time 
import json
import polars as pl
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
# See logging https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/search/azure-search-documents
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

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"]
key = os.environ["OPENAI_API_KEY"]


kernel = sk.Kernel()

#deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_chat_service(
    "chat_completion", AzureChatCompletion(
        deployment_name=deployment_name, 
        endpoint=endpoint, 
        api_key=key)
)

skill = kernel.import_semantic_skill_from_directory("samples/skills", "FunSkill")
joke_function = skill["Joke"]
print(joke_function("soccer in Argentina"))