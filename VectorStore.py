import os

from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake

load_dotenv()

def insert_deeplake(texts=None,activeloop_org_id=None,embeddings=None):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs = text_splitter.create_documents(texts=texts)
    activeloop_dataset_name = "langchain_couse_0_hero"
    dataset_path = f"hub://{activeloop_org_id}/{activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path,embedding_function=embeddings)
    db.add_documents(docs)

    return db

def retrieve(llm=None,database = None,question="Which year was Vasu born?"):
    retrieval_qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=database.as_retriever())
    tools = [
        Tool(name="Retrieval QA system",func=retrieval_qa.run,description="Useful for answering questions")
    ]

    agent = initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
    response = agent.run(question)
    print(response)

if __name__ == "__main__":
    activeloop_org_id = os.getenv("ACTIVELOOP_ORG_ID")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    #texts = ["Vasu never liked the sticky part of a banana.","Vasu was born on 12th October, 1988."]
    texts = ["If yesterday is equal to Today, then today must be equal to tomorrow","Plot no 28 belongs to Maitri appt"]
    db = insert_deeplake(texts=texts,activeloop_org_id=activeloop_org_id,embeddings=embeddings)

    #question = "What does Vasu dislike?"
    question = "Which plot belongs to Maitri Apartments?"
    llm = OpenAI(model_name="text-davinci-003", temperature=0.2)
    retrieve(llm=llm,database=db,question=question)
