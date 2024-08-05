#Import necessary packages
    #python.exe -m pip install --upgrade pip
    #pip install streamlit
    #pip install langchain
    #pip install faiss-gpu or pip install faiss-cpu
    #pip install openai langchain (Falcon AI71)
    #pip install LLMChain


    #pip install tiktoken (explicit installation not required in latest version langchain)
    ####pip install pydantic (explicit installation not required in latest version langchain)

    

import streamlit as st

from collections import deque
from typing import Dict, List, Optional

from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore

#from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel, Field
from langchain_community.embeddings import JinaEmbeddings

#'''LangChain SDK '''
from langchain.chat_models import ChatOpenAI



#Step 1 - Chain 1 - Create TaskCreationChain
class TaskCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            partial_variables={"objective": objective},
            input_variables=["result", "task_description", "incomplete_tasks"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def get_next_task(
        self, result: Dict, task_description: str, task_list: List[str]
    ) -> List[Dict]:
        """Get the next task."""
        incomplete_tasks = ", ".join(task_list)
        response = self.run(
            result=result,
            task_description=task_description,
            incomplete_tasks=incomplete_tasks,
        )
        new_tasks = response.split("\n")
        return [
            {"task_name": task_name} for task_name in new_tasks if task_name.strip()
        ]


#Step 2 - Chain 2 - Create TaskPrioritizationChain
class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            partial_variables={"objective": objective},
            input_variables=["task_names", "next_task_id"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def prioritize_tasks(self, this_task_id: int, task_list: List[Dict]) -> List[Dict]:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in task_list]
        next_task_id = int(this_task_id) + 1
        response = self.run(task_names=task_names, next_task_id=next_task_id)
        new_tasks = response.split("\n")
        prioritized_task_list = []
        for task_string in new_tasks:
            if not task_string.strip():
                continue
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                prioritized_task_list.append(
                    {"task_id": task_id, "task_name": task_name}
                )
        return prioritized_task_list

#Step 3 - Chain 3 - Create ExecutionChain
class ExecutionChain(LLMChain):
    """Chain to execute tasks."""

    vectorstore: VectorStore = Field(init=False)

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = True
    ) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "You are an AI who performs one task based on the following objective: {objective}."
            " Take into account these previously completed tasks: {context}."
            " Your task: {task}."
            " Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose, vectorstore=vectorstore)

    def _get_top_tasks(self, query: str, k: int) -> List[str]:
        """Get the top k tasks based on the query."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return []
        sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
        return [str(item.metadata["task"]) for item in sorted_results]

    def execute_task(self, objective: str, task: str, k: int = 5) -> str:
        """Execute a task."""
        context = self._get_top_tasks(query=objective, k=k)
        return self.run(objective=objective, context=context, task=task)


#Step 4 - Combine above 3 chains created - Controller Model for the BabyAGI agent
class BabyAGI(BaseModel):
    """Controller model for the BabyAGI agent."""
    print("Inside Controller model for the BabyAGI agent.")
    objective: str = Field(alias="objective")
    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: ExecutionChain = Field(...)
    task_id_counter: int = Field(1)

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        st.text("Task List ⏰")
        for t in self.task_list:
            st.write("- " + str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        st.subheader("Next Task:")
        st.warning("- " + str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        st.subheader("Task Result")
        st.info(result, icon="ℹ️")

    def print_task_ending(self):
        st.success("Tasks terminated.", icon="✅")

    def run(self, max_iterations: Optional[int] = None):
        """Run the agent."""
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = self.execution_chain.execute_task(
                    self.objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.execution_chain.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = self.task_creation_chain.get_next_task(
                    result, task["task_name"], [t["task_name"] for t in self.task_list]
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    self.task_prioritization_chain.prioritize_tasks(
                        this_task_id, list(self.task_list)
                    )
                )
            num_iters += 1
            if max_iterations is not None and num_iters == max_iterations:
                self.print_task_ending()
                break

    @classmethod
    def from_llm_and_objectives(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        objective: str,
        first_task: str,
        verbose: bool = False,
    ) -> "BabyAGI":
        
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, objective, verbose=verbose
        )
        print("task_creation_chain success !!!!!!!!!!", type(task_creation_chain))
        
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, objective, verbose=verbose
        )
        print("task_prioritization_chain success !!!!!!!!!!", type(task_prioritization_chain))

        execution_chain = ExecutionChain.from_llm(llm, vectorstore, verbose=verbose)
        print("execution_chain success !!!!!!!!!!", type(execution_chain))

        controller = cls(
            objective=objective,
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
        )
        print("controller success !!!!!!!!!!", type(controller))
        controller.add_task({"task_id": 1, "task_name": first_task})
        return controller


def initial_embeddings(embedding_api_key, first_task):
    with st.spinner("Initial Embeddings ... "):
        
        # Define your embedding model

        #'''Jina Embeddings : START'''
        embeddings = JinaEmbeddings(jina_api_key=embedding_api_key, model_name="jina-embeddings-v2-base-en")
        print("embeddings ::: ", embeddings)
        #'''Jina Embeddings : END'''


        vectorstore = FAISS.from_texts(
            ["_"], embeddings, metadatas=[{"task": first_task}]
        )
    return vectorstore



# 5. Streamlit page 
#st.title("👶🏼 Baby-AGI 🤖 ")
st.title("👶🏼Falcon Autonomous Agent 🤖")
st.subheader(" Powered by Falcon")

# 6. SHARED ON ALL PAGES : Initialize session states
if "jinaEmbedAPIkey" not in st.session_state:
        st.session_state["jinaEmbedAPIkey"] = ""

jina_embedding_api_key = st.text_input("Jina Embedding API-KEY", type="password", key="jinaEmbedAPIkey")


# 7. API - Ask the user to enter their Falcon API key
#AI71_BASE_URL = "https://api.ai71.ai/v1/"

API_O = st.session_state["apikey"]
api_url = st.session_state.apiURL

max_iterations = st.session_state["promptNo"]

print("api_url AGENT 111111111 :::::::::::::::::::::::::::::::::::", api_url)

#jina_embedding_api_key = "jina_0ebca2c02a0f45c08c5b7611175a5f5evjTbCjf2warA_2CAqWGwfI-6boCL"
#jina_embedding_api_key = st.session_state.jinaEmbedAPIkey

print("jina_embedding_api_key :::::::::::::::::::::::", jina_embedding_api_key)

if API_O:
        OBJECTIVE = st.text_input(
            label=" 🏁 :orange[What's Your Ultimate Goal]: ",
            value="Learn Python in 3 days",
        )

        # first_task = "Make a todo list"

        first_task = st.text_input(
            label=" 🥇:range[WInitial task:] ",
            value="Make a todo list",
        )

        embedding_api_key = jina_embedding_api_key

        if embedding_api_key:
            vectorstore = initial_embeddings(embedding_api_key, first_task)

            if st.button(" 🪄 Let me perform the magic 👼🏼"):
                try:
                    print("Let me perform the magic............................")
                    #'''Falcon llm : START'''
                    llm = ChatOpenAI(
                        model="tiiuae/falcon-180B-chat",
                        #model = MODEL,
                        api_key=API_O,
                        base_url=api_url,
                        streaming=True,
                    )
                    print("llm ::::::::::::::::::::::::::::::::::::::::::::::", llm)
                    #'''Falcon llm : END'''



                    baby_agi = BabyAGI.from_llm_and_objectives(
                        llm=llm,
                        vectorstore=vectorstore,
                        objective=OBJECTIVE,
                        first_task=first_task,
                        verbose=False,
                    )
                    print("baby_agi success !!!!!!!!!!!!!!", type(baby_agi))
                    
                    with st.spinner("👶 BabyAGI 🤖 at work ..."):
                        baby_agi.run(max_iterations=max_iterations)

                    st.balloons()
                except Exception as e:
                    #st.error(e)
                    st.error(f'API Key Error. Please check your API Key: {e.__class__}')
        else:
            st.warning("Jina Embedding is also necessary to use Autonomous AI Agent. The API key(s) are not stored in any form.")

else:
    st.warning("Falcon API KEY and Jina Embedding is necessary to use Autonomous AI Agent. The API key(s) are not stored in any form.")
