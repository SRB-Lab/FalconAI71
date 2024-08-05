import streamlit as st

AI71_BASE_URL = "https://api.ai71.ai/v1/"

def main():
    # 1. Set Streamlit page configuration
    st.set_page_config(page_title='🧠sAIsynergy - Falcon AI Assistant 🤖', layout='wide')

    # 2. --- PAGE SETUP ---
    about_page = st.Page(
        "views/sAIsynergy_about.py",
        title="About sAIsynergy",
        #icon=":material/account_circle:",
        icon=":material/smart_toy:",
        default=True,
    )
    project_1_page = st.Page(
        "views/sAIsynergy_memoryBot.py",
        title="Conversational Chat Bot",
        #icon=":material/bar_chart:",
        icon=":material/smart_toy:",
    )
    project_2_page = st.Page(
        "views/sAIsynergy_aiAgent.py",
        title="Autonomous AI Agent",
        icon=":material/smart_toy:",
    )

    # 3.--- NAVIGATION SETUP [WITH SECTIONS] ---
    pg = st.navigation(
        {
            "Info": [about_page],
            "Projects": [project_1_page, project_2_page],
        }
    )

    # 4. SHARED ON ALL PAGES : Initialize session states
    print("st.session_state in AI MAIN :::::::::::::::::::", st.session_state)
    
    #user_input = st.session_state["input"]
    #print("USER INPUT FOROM st.session_state in AI MAIN :::::::::::::::::::", user_input)

    if "generated" not in st.session_state:
        st.session_state["generated"] = [] #output
    if "past" not in st.session_state:
        st.session_state["past"] = [] #past
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "u_input" not in st.session_state:
        st.session_state["u_input"] = ""
    if "stored_session" not in st.session_state:
        st.session_state["stored_session"] = []
    if "apikey" not in st.session_state:
        st.session_state["apikey"] = ""
    if "modelSel" not in st.session_state:
        st.session_state["modelSel"] = ""
    if "promptNo" not in st.session_state:
        st.session_state["promptNo"] = ""
    if "apiURL" not in st.session_state:
        st.session_state["apiURL"] = ""
    

    # 5. API - Ask the user to enter their Falcon API key
    API_O = st.sidebar.text_input("Falcon AI71 API-KEY", type="password", key="apikey")
    
  
    # 6. Set up sidebar with various options
    with st.sidebar.expander("🛠️ View More Options", expanded=True):
        MODEL = st.selectbox(label='Model', options=['tiiuae/falcon-180B-chat', 'tiiuae/falcon-11b'])
        K = st.number_input(' (#)Summary of prompts to consider', min_value=3, max_value=10)

        st.session_state["modelSel"] = MODEL
        st.session_state["promptNo"] = K
        st.session_state["apiURL"] = AI71_BASE_URL
        
        
    

    # --- RUN NAVIGATION ---
    pg.run()

if __name__ == "__main__":
    main()

