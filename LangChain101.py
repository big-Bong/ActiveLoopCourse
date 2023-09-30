from dotenv import load_dotenv
from langchain.chains import ConversationChain, LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()
llm = OpenAI(model_name = "text-davinci-003", temperature=0.9)

def testLLM():
        text = "You are a personal trainer. Suggest a personalized workout routine for somone looking to improve \
                cardiovascular endurance, reduce weight and prefers outdoor activities."
        print(llm(text))

def testChain():
        template = "Suggest a funny name for a company selling {product_name} in China."
        prompt = PromptTemplate(input_variables=["product_name"],template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        print(chain.run("sex toys"))

def testMemory():
        conversation = ConversationChain(llm=llm,verbose=True,memory=ConversationBufferMemory())
        conversation.predict(input="Remember the phone number 98**98**")
        conversation.predict(input="What all can you do?")
        conversation.predict(input="Can you tell me the temperature in Delhi today?")
        conversation.predict(input="What is my name and what was the phone number?")
        conversation.predict(input="")

        print(conversation)

if __name__ == "__main__":
        testLLM()
        #testChain()
        #testMemory()