from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain_classic.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# --- 1. Documents and splitting -----------------------------------------
# raw documents go here, then split into chunks

RAW_DOC = [
    """Employees working remotely must use company-managed devices and connect through the approved VPN when accessing internal systems. Remote employees are responsible for maintaining a secure workspace and must not share company information with unauthorized individuals."""
    
    """VPN access is required whenever employees access confidential company resources from outside the corporate network. Employees must use their assigned credentials, enable multi-factor authentication, and report any suspected unauthorized access or credential compromise to IT immediately."""
    
    """All new employees must complete the IT onboarding process during their first week. This includes setting up company accounts, configuring required software, completing security training, and reviewing the organization's acceptable-use and data-protection policies."""
    
    """Company devices must be protected with strong passwords, automatic screen locking, and approved security software. Employees must install security updates when requested by IT and should not install unauthorized applications or connect unknown USB devices to company equipment."""
    ]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.create_documents(RAW_DOC)

print(f"Number of chunks: {len(chunks)}")

"""for i, chunk in enumerate(chunks):
    print(f"--- chunk {i} ---")
    print(chunk.page_content)"""


# --- 2. Embeddings --------------------------------------------------------
# custom TF-IDF based Embeddings class goes here

class CustomEmbeddings(Embeddings):

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.is_fitted = False

    def embed_documents(self, texts):
        vectors = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return vectors.toarray().tolist()

    def embed_query(self, text):
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before embedding a query.")

        vector = self.vectorizer.transform([text])
        return vector.toarray()[0].tolist()


# --- 3. Vector store and retriever ----------------------------------------
# FAISS vector store built from chunks + embeddings, then .as_retriever()

embeddings = CustomEmbeddings()

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

"""results = retriever.invoke("what are the VPN requirements")
for r in results:
    print(r.page_content)"""


# --- 4. Tools --------------------------------------------------------------
# retriever wrapped as a tool, plus calculator and ticket lookup tools

retriever_tool = create_retriever_tool(
    retriever,
    "search_it_policies",
    "Search the company's IT policies to answer questions about remote work, VPN access, employee onboarding, device security, and other internal IT rules."
)

@tool
def calculator(expression: str) -> str:
    """Use this tool when you need to perform arithmetic calculations such as addition, subtraction, multiplication, division, or percentages."""
    
    allowed = "0123456789+-*/().% "
    
    if not all(char in allowed for char in expression):
        return "Invalid expression. Only basic arithmetic is allowed."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Invalid arithmetic expression."

@tool
def check_ticket(ticket_id: str) -> str:
    """Use this tool when you need to check the status of an IT support ticket using its ticket ID."""

    tickets = {
        "IT-1001": "Open - IT team is investigating the issue.",
        "IT-1002": "In Progress - Waiting for the user's response.",
        "IT-1003": "Resolved - The issue has been fixed."
    }

    ticket_id = ticket_id.strip().upper()

    if ticket_id.isdigit():
        ticket_id = f"IT-{ticket_id}"

    return tickets.get(ticket_id, f"Ticket {ticket_id} was not found.")

tools = [retriever_tool, calculator, check_ticket]


# --- 5. Model ----------------------------------------------------------------
# chat model setup goes here 

#model = init_chat_model("google_genai:gemini-3.8-flash")
model = init_chat_model(
    "google_genai:gemini-3.5-flash-lite"
)

# --- 6. Agent ------------------------------------------------------------------
# create_agent(...) with tools, system prompt, and a checkpointer for memory

checkpointer = InMemorySaver()

system_prompt = """
You are an internal IT support assistant.

Help employees with IT policies, calculations, and support ticket status.
Use the available tools whenever they are relevant. For company policy
questions, always use the policy search tool instead of guessing or relying
on general knowledge.

If a tool does not provide the required information, clearly say that you
could not find the information. Do not invent company policies or ticket
details.

If a question is unrelated to IT policy, calculations, or ticket status, politely explain that it's outside what you can help with.
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=checkpointer
)

# --- 7. Run --------------------------------------------------------------------

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)

def ask(question, thread_id="demo-thread"):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = agent.invoke(
        {
            "messages": [HumanMessage(content=question)]
        },
        config=config
    )

    return extract_text(result["messages"][-1].content)



if __name__ == "__main__":

    """# Test 1: Policy search
    print("Policy:")
    print(ask("What is the company policy for using VPN when working remotely?"))

    # Test 2: Calculator
    print("\nCalculator:")
    print(ask("What is 125 * 8 + 50?"))

    # Test 3: Ticket lookup
    print("\nTicket:")
    print(ask("What is the status of ticket IT-1001?"))

    # Test 4 & 5: Memory test
    print("\nMemory test:")
    print(ask(
        "What is the company policy regarding VPN access?",
        thread_id="memory-test"
    ))

    print(ask(
        "Does that also apply to onboarding?",
        thread_id="memory-test"
    ))"""

    print("Internal IT Support Assistant (type 'exit' to quit)")
    thread_id = "interactive-session"

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer = ask(question, thread_id=thread_id)
        print(f"Assistant: {answer}")