import os
import openai
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
#read .env file
from dotenv import load_dotenv
load_dotenv()

# os.environ["OPENAI_API_KEY"] #= os.getenv('OPENAI_API_KEY')
# os.environ["AZURE_OPENAI_ENDPOINT"] #= os.getenv('AZURE_OPENAI_ENDPOINT')

def OpenAIembeddings():
    open_ai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",
        chunk_size=10,
    )
    return open_ai_embeddings
    
def llm():
    return AzureChatOpenAI(model_name="gtp35turbo-latest")


