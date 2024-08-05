#Install dependancy modules
    #pip install streamlit
    #pip install langchain
    #pip install langchain_community
    #pip pip install openai langchain 

import streamlit as st

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

#AI71 - LangChain SDK
from langchain.chat_models import ChatOpenAI



# 1 . Set up the Streamlit app layout
st.title("🤖 Conversational Chat Bot with 🧠")
st.subheader(" Powered by Falcon LLM ")
#st.subheader(" LangChain + Streamlit ")

# 2 . Read required data from session 
modelSelected = st.session_state["modelSel"]
promptNum = st.session_state["promptNo"]
api_url = st.session_state["apiURL"]
API_O = st.session_state["apikey"]

# 3. Define function to get user input
def get_text():
    """ Get the user input text. Returns: (str): The text entered by the user """
    input_text = st.text_input(label="You: ", key="u_input", value=st.session_state.input, placeholder="Your AI assistant here! Ask me anything ...", label_visibility='hidden')
    return input_text


# 4. Clear user input text
def clear_input():
    st.session_state["u_input"] = ""
    st.session_state.u_input = ""
    st.session_state.input =""
    st.session_state["input"] = ""

# 5. Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = [] #output
    st.session_state["past"] = [] #past
    st.session_state["input"] = ""
    st.session_state["u_input"] = ""




# 6. If API key available, get user input upon valid API Key. Session state storage would be ideal
user_input = ""
if API_O:
    # 6.1 Create an Gen AI llm instance
    print("Creating ChatOpenAI falcon chain........................................", modelSelected)
    #st.write("API Key Validation")

    print("api_url :::::::::::::::::::::::", api_url)

    try:
        llm = ChatOpenAI(
            model = modelSelected,
            api_key=API_O,
            base_url=api_url,
            streaming=True,
        )       

        #st.write("Successful !!")

        # 6.2 Create a ConversationEntityMemory object if not already created
        if 'entity_memory' not in st.session_state:
                st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=promptNum)
            
        # 6.3 Create the ConversationChain object with the specified configuration
        Conversation = ConversationChain(
                llm=llm, 
                prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                memory=st.session_state.entity_memory
            )
        
        # 6.4. Render user input widget
        user_input = get_text()

        # 6.5. Clear button if there is text input widget enabled
        st.button('Clear Input Text', on_click=clear_input)

        # 6.6. Add a button to start a new chat
        st.sidebar.button("New Chat", on_click=new_chat, type='primary') #new_chat function call

    except Exception as e:
        print("AN EXCEPTION OCCURED", e.__class__)
        st.error(f'API Key Error. Please check your API Key: {e.__class__}')
else:
    st.error('Falcon LLM model API key required to start Conversational Chat Bot. The API key is not stored in any form.')

             

    
# 7. Generate the output using the ConversationChain object and the user input, and add the input/output to the session
print("USER INPUT iN MEM BOT ::::", user_input)
if user_input:
    output = Conversation.run(input=user_input)  
    st.session_state.past.append(user_input)  
    st.session_state.generated.append(output)

# 8. Allow to download as well
download_str = []

# 8.1. Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i]) #part of step 9
        download_str.append(st.session_state["generated"][i]) #part of step 9

    # 8.2. Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download', download_str, on_click=clear_input)



# 9. Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label= f"Conversation-Session:{i}"):
        st.write(sublist)

# 10. Allow the user to clear all stored conversation sessions
def clearHistory():
    del st.session_state.stored_session

if st.session_state.stored_session:   
    st.sidebar.button("Clear-History", on_click=clearHistory)


