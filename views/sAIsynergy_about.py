#Install dependancy modules
    #pip install streamlit

import streamlit as st

st.write("\n")
st.subheader("About", anchor=False)
st.write(
    """
    - sAIsynergy - A Falcon AI assistant developed by sAIexperiments
    - Powered by Falcon LLM (developed by the Technology Innovation Institute (TII) in Abu Dhabi, is a leader in generative AI)
    - Can be used as an enterprise AI app or as a personal assistant app that integrates advanced AI capabilities tailored to specific needs and preferences. 
    """
)

# --- SKILLS ---
st.write("\n")
st.subheader("Specification", anchor=False)
st.write(
    """
    - AI71 Falcon LLM [tiiuae/falcon-180B-chat, tiiuae/falcon-11b] 
    - LangChain
    - Streamlit 
    - Faiss (cpu or gpu) 
    - Python
    """
)

API_O = st.session_state["apikey"]
if API_O:
    print()
else:
    st.subheader("API Key Required", anchor=False)
    st.write(
    """
    - Valid Falcon API Key (to use Conversational Chat Bot)
    - Valid Jina Embedding API KEY (to use Autonomous AI Agent)
    """
)
    