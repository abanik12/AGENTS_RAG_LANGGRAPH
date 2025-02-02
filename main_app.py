from typing import Literal
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

tools = [retrieve_context, web_search]
tool_node = ToolNode(tools)

llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18").bind_tools(tools)    


# Function that invokes the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}  # Returns as a list to add to the state

def should_continue(state:MessagesState):
    messages=state['messages']
    last_message=messages[-1] 
    if last_message.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)

builder.add_node("llm_node",call_model)
builder.add_node("tools",tool_node)

builder.add_edge(START,"llm_node")
builder.add_conditional_edges("llm_node",should_continue)
builder.add_edge("tools","llm_node")

memory = MemorySaver()
react_graph=builder.compile(checkpointer=memory)

more_questions = True
while(more_questions):
    THREAD_ID = "11"
    config = {"configurable":{"thread_id":THREAD_ID}}

    question = input("Ask your question : ")
    if question == "":
        more_questions=False

    response = react_graph.invoke({"messages": [HumanMessage(content=question)]},config)

    for m in response["messages"]:
        m.pretty_print()
