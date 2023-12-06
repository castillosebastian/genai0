# -*- coding: utf-8 -*-
"""zephyr-7b-beta-feature-test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UoPcoiA5EOBghxWKWduQhChliMHxla7U

Note: Responses from local models can be quite slow, especially with 8-bit quantization.

With 4bit quantization, `HuggingFaceH4/zephyr-7b-beta` uses about 8GB of VRAM and spiked to 14GB of RAM when loading the model, then settled around 5GB. I used a T4 instance for this notebook.
"""

!pip install llama-index transformers accelerate bitsandbytes

"""## Setup

### Data
"""

from llama_index.readers import BeautifulSoupWebReader

url = "https://www.theverge.com/2023/9/29/23895675/ai-bot-social-network-openai-meta-chatbots"

documents = BeautifulSoupWebReader().load_data([url])

"""### LLM

This should run on a T4 instance on the free tier
"""

import torch
from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt


llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)

from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")

"""### Index Setup"""

from llama_index import VectorStoreIndex

vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)

from llama_index import SummaryIndex

summary_index = SummaryIndex.from_documents(documents, service_context=service_context)

"""### Helpful Imports / Logging"""

from llama_index.response.notebook_utils import display_response

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""## Basic Query Engine

### Compact (default)
"""

query_engine = vector_index.as_query_engine(response_mode="compact")

response = query_engine.query("How do OpenAI and Meta differ on AI tools?")

display_response(response)

"""### Refine"""

query_engine = vector_index.as_query_engine(response_mode="refine")

response = query_engine.query("How do OpenAI and Meta differ on AI tools?")

display_response(response)

"""### Tree Summarize"""

query_engine = vector_index.as_query_engine(response_mode="tree_summarize")

response = query_engine.query("How do OpenAI and Meta differ on AI tools?")

display_response(response)

"""## Router Query Engine"""

from llama_index.tools import QueryEngineTool, ToolMetadata

vector_tool = QueryEngineTool(
    vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for specific facts."
    )
)

summary_tool = QueryEngineTool(
    summary_index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarizing an entire document."
    )
)

"""### Single Selector"""

from llama_index.query_engine import RouterQueryEngine

query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool],
    service_context=service_context,
    select_multi=False
)

response = query_engine.query("What was mentioned about Meta?")

display_response(response)

"""### Multi Selector"""

from llama_index.query_engine import RouterQueryEngine

query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool],
    service_context=service_context,
    select_multi=True,
)

response = query_engine.query("What was mentioned about Meta? Summarize with any other companies mentioned in the entire document.")

display_response(response)

"""## SubQuestion Query Engine"""

from llama_index.tools import QueryEngineTool, ToolMetadata

vector_tool = QueryEngineTool(
    vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for specific facts."
    )
)

summary_tool = QueryEngineTool(
    summary_index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarizing an entire document."
    )
)

import nest_asyncio
nest_asyncio.apply()

from llama_index.query_engine import SubQuestionQueryEngine

query_engine = SubQuestionQueryEngine.from_defaults(
    [vector_tool, summary_tool],
    service_context=service_context,
    verbose=True,
)

response = query_engine.query("What was mentioned about Meta? How Does it differ from how OpenAI is talked about?")

display_response(response)

"""## SQL Query Engine

Here, we download and use a sample SQLite database with 11 tables, with various info about music, playlists, and customers. We will limit to a select few tables for this test.
"""

import locale
locale.getpreferredencoding = lambda: "UTF-8"

!curl https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip -O /content/chinook.zip
!unzip /content/chinook.zip

from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, column

engine = create_engine("sqlite:////content/chinook.db")

from llama_index import SQLDatabase

sql_database = SQLDatabase(engine)

from llama_index.indices.struct_store import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["albums", "tracks", "artists"],
    service_context=service_context
)

response = query_engine.query("What are some albums? Limit to 5.")

display_response(response)

response = query_engine.query("What are some artists? Limit it to 5.")

display_response(response)

"""This last query should be a more complex join"""

response = query_engine.query("What are some tracks from the artist AC/DC? Limit it to 3")

display_response(response)

print(response.metadata['sql_query'])

"""## Programs

Depending the LLM, you will have to test with either `OpenAIPydanticProgram` or `LLMTextCompletionProgram`
"""

from typing import List
from pydantic import BaseModel

from llama_index.program import OpenAIPydanticProgram, LLMTextCompletionProgram

class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]

from llama_index.output_parsers import PydanticOutputParser

prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
program = LLMTextCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(Album),
    prompt_template_str=prompt_template_str,
    llm=llm,
    verbose=True,
)

output = program(movie_name="The Shining")

print(output)

"""## Data Agent

Similar to programs, OpenAI LLMs will use `OpenAIAgent`, while other LLMs will use `ReActAgent`.
"""

from llama_index.agent import OpenAIAgent, ReActAgent

agent = ReActAgent.from_tools(
    [vector_tool, summary_tool],
    llm=llm,
    verbose=True
)

"""Some inputs are hallucinated, causing issues with responses. Likely a better system prompt or tool descriptions could help."""

response = agent.chat("Hello!")
print(response)

response = agent.chat("What was mentioned about Meta? How Does it differ from how OpenAI is talked about?")
print(response)

