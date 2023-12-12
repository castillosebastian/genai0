# -*- coding: utf-8 -*-
'''
Model used:
- 7b-full: https://huggingface.co/ChanceFocus/finma-7b-full/tree/main
todo: 
    -30b : https://huggingface.co/ChanceFocus/finma-30b-nlp/tree/main 
        135 GB only model 
'''

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

"""## Setup LLM
work with FinMa

"""

#!pip install llama-index
import llama_index

model_url = "https://huggingface.co/ChanceFocus/finma-7b-full/tree/main"

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

"""We can tell that the model is using `metal` due to the logging!
## Start using our `LlamaCPP` LLM abstraction!
We can simply use the `complete` method of our `LlamaCPP` LLM abstraction to generate completions given a prompt.
"""

response = llm.complete("Hello! Can you tell what is love for Socrates")
print(response.text)

"""We can use the `stream_complete` endpoint to stream the response as it’s being generated rather than waiting for the entire response to be generated."""

response_iter = llm.stream_complete("What is democracy for Artistoteles?")
for response in response_iter:
    print(response.delta, end="", flush=True)

"""## Query engine set up with LlamaCPP
We can simply pass in the `LlamaCPP` LLM abstraction to the `LlamaIndex` query engine as usual.
But first, let's change the global tokenizer to match our LLM.
"""

from llama_index import set_global_tokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM

set_global_tokenizer(
    LlamaTokenizer.from_pretrained('ChanceFocus/finma-7b-full')
)

# use Huggingface embeddings
from llama_index.embeddings import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# create a service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

# load documents

### Get data
import requests
import os
dirpath = 'related_works/Cloud_VM/'
filename = dirpath + 'ey.pdf'
url = 'https://assets.ey.com/content/dam/ey-sites/ey-com/nl_nl/topics/jaarverslag/downloads-pdfs/2022-2023/ey-nl-financial-statements-2023-en.pdf'

if not os.path.exists(filename):
    print(f"Downloading {filename} from {url}...")    
    response = requests.get(url)
    with open(dirpath + 'ey.pdf', 'wb') as f:
        f.write(response.content)

documents = SimpleDirectoryReader(
    input_files=[filename]
).load_data()

# create vector store index
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# set up query engine
query_engine = index.as_query_engine()

response = query_engine.query("Summarise the Consolidated statement of cash flows of Ernst & Young Nederland LLP for 2022 and 2023")
print(response)
# Baja calidad
response = query_engine.query("Can you explain the EY risk management framework")
print(response)
# Calidad aceptable
 