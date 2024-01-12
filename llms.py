#gpt-3.5-turbo-instruct

# Load environment variables
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())


from langchain.llms import OpenAI
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
llm("explain large language models in one sentence")


#C:\Users\Mose\Documents\LangChain\venv\Scripts\spyder.exe





