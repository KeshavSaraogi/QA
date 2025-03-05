import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking

os.environ["LANGCHAIN_API_KEY"]         = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]      = "true"
os.environ["LANGCHAIN_PROJECT"]         = "Q&A Chatbot With Ollama"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please respond to the user queries.",
        ),
        ("user", "Question:{question}"),
    ]
)

def generate_response(question, model_name, temperature):
    llm                 = Ollama(model=model_name, temperature=temperature)
    output_parser       = StrOutputParser()
    chain               = prompt | llm | output_parser
    answer              = chain.invoke({"question": question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot With Ollama")

## LLM Model and Parameters
llm = st.sidebar.selectbox("Select Open Source model", ["gemma:2b"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

## User input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, llm, temperature)
    st.write(response)

else:
    st.write("Please provide the user input")
