import os
import requests
from constants import GROQ_API_KEY
from langchain_groq import ChatGroq
from langchain import PromptTemplate 
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["GROQ_API_KEY"]=GROQ_API_KEY

# Streamlit framework - Set page config
st.set_page_config(
    page_title="Celebrity Info",  # Title of the web page
    page_icon=":star:",  # Icon in the browser tab
    layout="wide",  # Wide layout for more space
    initial_sidebar_state="collapsed",  # Initially hide the sidebar
)

# Add custom CSS to style the app
st.markdown("""
    <style>
        body {
            background-color: #F7F7F7;
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #4B4B4B;
            margin-top: 20px;
        }
        .subheader {
            font-size: 18px;
            color: #666;
        }
        .container {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 30px;
        }
        .info-card {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            width: 48%;
        }
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
        }
        .celebrity-image {
            border-radius: 10px;
            max-width: 80%;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }
        .info-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        }
        .expander {
            background-color: #F1F1F1;
            border-radius: 5px;
            padding: 15px;
        }
        .expander .st-expanderHeader {
            color: #4B4B4B;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit Header
st.markdown("<div class='header'>Celebrity Info</div>", unsafe_allow_html=True)
input_text= st.text_input("Name of the celebrity:")

def get_celebrity_image(name):
    try:
        # Wikipedia API URL
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('thumbnail', {}).get('source', None)  
        else:
            return None
    except Exception as e:
        return None

#Memory
person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory=ConversationBufferMemory(input_key='person',memory_key='dob_history')
events_memory=ConversationBufferMemory(input_key='dob',memory_key='events_history')

#Prompt Template
first_input_prompt=PromptTemplate(
    input_variable=['name'],
    template= "Tell me about the celebrity {name}"
)

#OPENAI LLMS
llm=ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.8
)
chain=LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)

#Prompt2 Template
second_input_prompt=PromptTemplate(
    input_variable=['person'],
    template= "When was {person} born"
)

#OPENAI LLMS
llm=ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.8
)
chain2=LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)

#Prompt3 Template
third_input_prompt=PromptTemplate(
    input_variable=['dob'],
    template= "Mention 5 major events occured around {dob} in the world"
)

#OPENAI LLMS
llm=ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.8
)
chain3=LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='events', memory=events_memory)

parent_chain=SequentialChain(
    chains=[chain,chain2,chain3], input_variables=['name'], output_variables=['person','dob','events'],verbose=True
)

output=None 
if input_text:
    output=parent_chain({'name':input_text})
    dob = output.get('dob','DOB not found')
    name=output.get('name','Name not found')
    
    # Fetch celebrity image
    image_url = get_celebrity_image(name)
    
    # Display celebrity image
    if image_url:
        st.markdown(
            f"""
            <div style="text-align: center;">
            <img src="{image_url}" alt="Image of {name}" style="max-width: 100%; border-radius: 10px;"/>
            <p style="font-weight: bold;">Image of {name}</p>
            </div>
            """,
            unsafe_allow_html=True
            )
    else:
        st.write("No image found for this celebrity.")
    
    #st.write(output)
    
    with st.expander('About Person'): 
        st.info(person_memory.buffer)
    
    with st.expander('Person DOB'): 
        st.info(dob)

    with st.expander('Major Events around DOB'): 
        st.info(events_memory.buffer)
    