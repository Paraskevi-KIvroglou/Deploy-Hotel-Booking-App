import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import langchain.globals
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

my_model_id = os.getenv('MODEL_REPO_ID', 'Default Value')
my_model_id = "KvrParaskevi/Hotel-Assistant-Attempt5-Llama-2-7b"

@st.cache_resource 
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(my_model_id)
    model = AutoModelForCausalLM.from_pretrained(my_model_id)
    
    return tokenizer,model

def demo_miny_memory(model):
    # llm_data = get_Model(hugging_face_key)
    memory = ConversationBufferMemory(llm = model,max_token_limit = 512)
    return memory

def demo_chain(input_text, memory,model):
    # llm_data = get_Model(hugging_face_key)
    llm_conversation = ConversationChain(llm=model,memory=memory,verbose=langchain.globals.get_verbose())

    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply
