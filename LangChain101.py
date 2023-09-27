import os

from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()

llm = OpenAI(model = "text-davinci-003", temperature=0.9)
text = "You are a personal trainer. Suggest a personalized workout routine for somone looking to improve \
        cardiovascular endurance, reduce weight and prefers outdoor activities."
print(llm(text))