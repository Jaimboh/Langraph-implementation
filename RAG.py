from dotenv import load_dotenv
from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
from langchain.prompts import PromptTemplate
import pprint
import os

# Load environment variables
load_dotenv()

run_local = 'No'
models = "openai"
openai_api_key = "Your_API_KEY"
google_api_key = "Your_API_KEY"
local_llm = 'Solar'
os.environ["TAVILY_API_KEY"] = ""

# Split documents
url = 'https://lilianweng.github.io/posts/2023-06-23-agent/'
loader = WebBaseLoader(url)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)
all_splits = text_splitter.split_documents(docs)

# Embed and index
if run_local == 'Yes':
    embeddings = GPT4AllEmbeddings()
elif models == 'openai':
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
else:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )

# Index
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()
print(retriever)

###################################################################

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """
    keys: Dict[str, any]

#############################################################

### Nodes ###

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    local = state_dict["local"]
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "local": local, "question": question}}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM Setup
    if run_local == "Yes":
        llm = ChatOllama(model=local_llm, temperature=0)
    elif models == "openai":
        llm = ChatOpenAI(
            model="gpt-4-0125-preview", 
            temperature=0, 
            openai_api_key=openai_api_key
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
            verbose=True,
        )

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """
    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    # LLM
    if run_local == "Yes":
        llm = ChatOllama(model=local_llm, temperature=0)
    elif models == "openai":
        llm = ChatOpenAI(
            model="gpt-4-0125-preview", 
            temperature=0, 
            openai_api_key=openai_api_key
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
            verbose=True,
        )
    
    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""
        score: str = Field(description="Relevance score 'yes' or 'no'")

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=grade)

    from langchain_core.output_parsers import JsonOutputParser
    parser = JsonOutputParser(pydantic_object=grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} 
        If the document contains keywords related to the user question, grade it as relevant. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. 
        Provide the binary score as a JSON with no preamble or explanation and use these instructions to format the output: {format_instructions}""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    # Score
    filtered_docs = []
    search = "No"  # Default do not opt for web search to supplement retrieval
    for d in documents:
        score = chain.invoke(
            {
                "question": question,
                "context": d.page_content,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes"  # Perform web search
            continue

    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "local": local,
            "run_web_search": search,
        }
    }

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. 
        Look at the input and try to reason about the underlying semantic intent / meaning. 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Provide an improved question without any preamble, only respond with the updated question: """,
        input_variables=["question"],
    )

    # Grader
    # LLM
    if run_local == "Yes":
        llm = ChatOllama(model=local_llm, temperature=0)
    elif models == "openai":
        llm = ChatOpenAI(
            model="gpt-4-0125-preview", 
            temperature=0, 
            openai_api_key=openai_api_key
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
            verbose=True,
        )
    
    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {
        "keys": {"documents": documents, "question": better_question, "local": local}
    }

def web_search(state):
    """
    Web search based on the re-phrased question using Tavily API.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Web results appended to documents.
    """
    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    # Perform Web Search
    search = TavilySearchResults(k=2)
    web_results = search.results(query=question)
    web_docs = []
    for w in web_results:
        content = w["body"]
        doc = Document(page_content=content)
        web_docs.append(doc)

    return {
        "keys": {
            "documents": documents + web_docs,
            "question": question,
            "local": local,
            "web_docs": web_docs,
        }
    }

### Create the graph ###

sg = StateGraph(GraphState)
sg.set_as_starter("retrieve")
sg.add_state("retrieve", "grade_documents", retrieve)
sg.add_state("grade_documents", "transform_query", grade_documents, branch_1_name="run_web_search", branch_1_state="web_search", check_field="local", check_value="Yes")
sg.add_state("transform_query", "retrieve", transform_query)
sg.add_state("web_search", "generate", web_search)
sg.add_state("grade_documents", "generate", grade_documents, branch_1_name="run_web_search", branch_1_state="web_search")
sg.add_state("generate", END, generate)
sg.plot()

# Test the graph
example_question = {
    "keys": {"question": "How do transformers work?", "local": "No"}
}
output = sg.run(example_question)
pprint.pprint(output)
