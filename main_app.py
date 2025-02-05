from typing import TypedDict, Annotated
import operator
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv("./.env")

# defining agent state
class AgentState(TypedDict):
   messages: Annotated[list[AnyMessage], operator.add]


@tool
def retrieve_context(query: str):
    """Search for relevant context from the internal Tao Tanso company policy docs based on the input string passed"""

    pdf_path="/Users/relanto/AI WORK/RAG_ASSISTANT/documents/"
    loader=PyPDFDirectoryLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="company_docs",
        embedding=OpenAIEmbeddings(),
    )

    retriever = vectorstore.as_retriever()
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])

def web_search(query: str):
    """ Function to search the web for any general query other than internal company policy related queries"""
    tool=TavilySearchResults(max_results=2)
    answer=tool.invoke(query)
    return answer

toolset = [retrieve_context, web_search] 


class AssistantAgent:
    # initialising the object
    def __init__(self, model, tools, system_prompt = ""):
        self.system_prompt = system_prompt
        # initialising graph with a state 
        builder = StateGraph(AgentState)
        tool_node = ToolNode(tools)

        builder.add_node("llm_node",self.call_llm)
        builder.add_node("tools",tool_node)

        builder.add_edge(START,"llm_node")
        builder.add_conditional_edges("llm_node",self.should_continue)
        builder.add_edge("tools","llm_node")

        memory = MemorySaver()

        self.graph = builder.compile(checkpointer=memory)
        #self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_llm(self, state: AgentState):# Function that invokes the model
        messages = state['messages']
        # adding system prompt if it's defined
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        
        # calling LLM
        message = self.model.invoke(messages)
        return {'messages': [message]}
    
    def should_continue(self, state: AgentState):# Function that checks tool call return from llm
        messages=state['messages']
        last_message=messages[-1] 
        if last_message.tool_calls:
            return "tools"
        return END
    

# system prompt
prompt = '''You are a very helpful chatbot assistant for the Toyo Tanso Ltd.**, a multinational conglomerate involved in high-level research and manufacturing of carbon products with a goal of helping the employee with his questions. 
Your primary function is to help employees quickly by answering thier questions using precise, context-aware information from the companys knowledge base, which have documents related to compliance guidelines, project best practices, client onboarding materials and industry research reports.
The employee may also ask general question which needs to be answered using web search and should not reflect any data or answers from internal company policy documents '''


llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18")     
chat_agent = AssistantAgent(llm, toolset, prompt)  

more_questions = True
while(more_questions):
    THREAD_ID = "11"
    config = {"configurable":{"thread_id":THREAD_ID}}

    question = input("Ask your question : ")
    if question == "":
        more_questions=False
    messages = [HumanMessage(content=question)]

    for event in chat_agent.graph.stream({"messages": messages}, config):
        for v in event.values():
            v['messages'][-1].pretty_print()
