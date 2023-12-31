
import utils
import os
import requests
import llama_index
import torch
import llama_cpp

from llama_index import SimpleDirectoryReader
from llama_index import Document
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index import LLMPredictor

# Paramas
llama = True

### Get data
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


### Print data
print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])

### Create doc object
document = Document(text="\n\n".join([doc.text for doc in documents]))


### load model
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

if llama:
    # GPU
    from llama_cpp import Llama
    llm = None
    llm = Llama(
        model_path=model_path,
        n_threads=2, # CPU cores
        n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_gpu_layers=43, # Change this value based on your model and your GPU VRAM pool.
        n_ctx=4096, # Context window
    )    
else:
    from transformers import LlamaTokenizer, LlamaForCausalLM
    tokenizer = LlamaTokenizer.from_pretrained('ChanceFocus/finma-7b-full')
    llm = LlamaForCausalLM.from_pretrained('ChanceFocus/finma-7b-full', device_map='auto')


##### The replicate endpoint
from llama_index.llms import Replicate
from llama_index import ServiceContext, set_global_service_context
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
LLAMA_13B_V2_CHAT = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

# inject custom system prompt into llama-2
def custom_completion_to_prompt(completion: str) -> str:
    return completion_to_prompt(
        completion,
        system_prompt=(
            "You are a Q&A assistant. Your goal is to answer questions as "
            "accurately as possible is the instructions and context provided."
        ),
    )


llm = Replicate(
    model=LLAMA_13B_V2_CHAT,
    temperature=0.01,
    # override max tokens since it's interpreted
    # as context window instead of max tokens
    context_window=4096,
    # override completion representation for llama 2
    completion_to_prompt=custom_completion_to_prompt,
    # if using llama 2 for data agents, also override the message representation
    messages_to_prompt=messages_to_prompt,
)

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)
index = VectorStoreIndex.from_documents([document],
                                        service_context=service_context)

query_engine = index.as_query_engine()

response = query_engine.query(
    "What actions is Ernst & Young Global Limited taking to address climate change issues?"
)
print(str(response))

# ## Evaluation setup using TruLens

eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        print(item)
        eval_questions.append(item)

# You can try your own question:
new_question = "What is the right AI job for me?"
eval_questions.append(new_question)

print(eval_questions)

from trulens_eval import Tru
tru = Tru()

tru.reset_database()

from utils import get_prebuilt_trulens_recorder

tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                             app_id="Direct Query Engine")

with tru_recorder as recording:
    for question in eval_questions:
        response = query_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])

records.head()

# launches on http://localhost:8501/
tru.run_dashboard()

# ## Advanced RAG pipeline

# ### 1. Sentence Window retrieval

from utils import build_sentence_window_index

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)

from utils import get_sentence_window_query_engine

sentence_window_engine = get_sentence_window_query_engine(sentence_index)

window_response = sentence_window_engine.query(
    "how do I get started on a personal project in AI?"
)
print(str(window_response))

tru.reset_database()

tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id = "Sentence Window Query Engine"
)

for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))

tru.get_leaderboard(app_ids=[])

# launches on http://localhost:8501/
tru.run_dashboard()

# ### 2. Auto-merging retrieval

from utils import build_automerging_index

automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)

from utils import get_automerging_query_engine

automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)

auto_merging_response = automerging_query_engine.query(
    "How do I build a portfolio of AI projects?"
)
print(str(auto_merging_response))

tru.reset_database()

tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                                                         app_id="Automerging Query Engine")
for question in eval_questions:
    with tru_recorder_automerging as recording:
        response = automerging_query_engine.query(question)
        print(question)
        print(response)

tru.get_leaderboard(app_ids=[])

# launches on http://localhost:8501/
tru.run_dashboard()


