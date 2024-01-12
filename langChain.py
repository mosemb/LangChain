# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
#gpt-3.5-turbo-instruct

# Load environment variables
from dotenv import load_dotenv,find_dotenv
key = load_dotenv(r'C:\Users\Mose\Documents\Env\envf.env')
print(key)


from langchain.llms import OpenAI
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
print (llm("explain large language models in one sentence"))


#This is a chat model, 
# A chat model includes a schema, HumanMessage and a SystemMessages
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.chat_models import ChatOpenAI

# To use a chat model a SystemMessage and HumanMessage are used
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0.3)
messages = [SystemMessage(content="You are an expert data scientist"),
            HumanMessage(content="Write a python script that trains a neural network on simulated data")]

response = chat(messages)


print(response.content, end="\n")

#Prompt Templates 

#Prompt Templates, these are usually dynamic and they are going to be sent to the
#Language Model. These are not static but they are going to be dynamic. In that case
#We have prompt templates. These are going to be used in an application.
#We take a piece of text and inject that in the language model

from langchain import PromptTemplate
template = """ You are an expert data scientist with an expertise of building large language models. Explain the concept of {concept} in a couple of lines
           """
          
prompt = PromptTemplate(input_variables=["concept"], template=template)
print(prompt)

print (llm(prompt.format(concept="autoencoders")))


#Chains
#A chain takes a language model and a prompt template and combines them into 
# an interface that takes an imput from the user and outputs the answer from the
# Language model. We can build sequential chains where we get one chains output and the input for the other


from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run('autoencoders'))

second_prompt = PromptTemplate(input_variables=["ml_concept"],
                               template = "Turn the concept description of {ml_concept} and explain to me like am 5 in 500 words" )
chain_two = LLMChain(llm=llm, prompt=second_prompt)


from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain,chain_two], verbose=True)

explanation = overall_chain.run("autoencoder")
print(explanation) 

#Embeddings and Vector Stores - Store the data in Vector Stores
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
texts = text_splitter.create_documents([explanation])
print (texts)
print(" ")
print(texts[0])

# We can use the data that we have got to convert them into vector representations
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(deployment="ada")


query_result = embeddings.embed_query(texts[0].page_content)
print(query_result)



import os
import pinecone
from langchain.vectorstores import Pinecone

api_key=os.getenv('PINECONE_API_KEY')
environment=os.getenv('PINECONE_ENVIRONEMENT')

print(api_key)
print(environment)

#Initialize PineCone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  
    environment=os.getenv('PINECONE_ENVIRONEMENT')
              )

index_name = "langchainq"
#pinecone.create_index(index_name, dimension=1536,
#                          metric="cosine", pods=1, pod_type="p1.x1")
search = Pinecone.from_documents(texts, embeddings, index_name=index_name)

query = "What is magical about an encoder"
result=search.similarity_search(query)


print(result)

#Agents
#from langchain.agents.agent_toolkits import create_python_agent
#from langchain.tools.python.tool import PythonREPLTool
#from langchain.python import PythonREPL
#from langchain.llms.openai import OpenAI




 





              
              














#print(5*7)






