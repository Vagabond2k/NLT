from uuid import uuid4

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.tools import tool
from langchain.agents import create_agent
import ollama

# --- Embeddings & vector store ----------------------------------------------
embeddings = OllamaEmbeddings(
    model="all-minilm:l6-v2",
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)


# --- Documents (same as in minimal script) -----------------------------------
document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
    id=4,
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
    id=5,
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
    id=6,
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    id=7,
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    id=8,
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
    id=9,
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    id=10,
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)


# --- Tool: same retrieval as minimal, but wrapped for the agent -------------
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Search the local knowledge base (news, tweets, websites, etc.)
    and return the most relevant passages to answer the query.
    """
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    # The agent sends `serialized` back into the model as tool output.
    # `retrieved_docs` is there as an artifact if you want it programmatically.
    return serialized, retrieved_docs


# --- LLM: your gemma tools model --------------------------------------------
client = ollama.Client(
    host="http://127.0.0.1:11434",
    trust_env=False,  # <-- THIS is the important bit
)

llm = ChatOllama(
    client=client,                       # use our custom client
    model="llama3.1:latest",
    temperature=0.0,
    num_ctx=8042,
)


sys_prompt = """
You are a question-answering assistant with access to a retrieval tool called `retrieve_context`.

You MUST follow these rules:

1. Always call `retrieve_context` at least once before answering a question.
2. Base your final answer ONLY on the retrieved context. Do NOT use outside knowledge.
3. When the context contains specific numbers (like temperatures, dates, prices, or quantities),
   you MUST copy those numbers and units exactly as written. Do not convert or "correct" them.
4. Do NOT convert units (for example, do NOT convert Fahrenheit to Celsius or Celsius to Fahrenheit).
5. If the context does not contain enough information to answer the question, say:
   "I don't know based on the provided context."
"""

tools = [retrieve_context]
agent = create_agent(llm, tools, system_prompt=sys_prompt)


# --- Use the agent -----------------------------------------------------------
query = "What temperature will do tomorrow?"

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
