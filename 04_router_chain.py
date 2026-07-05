"""
04_router_chain.py
------------------
Concept: Demonstrate a Router Chain for dynamic decision-making.

Use Case: Smart content router for business messages.

LangChain Concept: RunnableBranch (conditional routing between different chains)
"""

import os
import json
from dotenv import load_dotenv

# must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough


# ==========================================
# 1️⃣ Setup
# ==========================================
load_dotenv()

with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
model_cfg = config[provider]

if provider == "openai":
    llm = ChatOpenAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.4),
        max_tokens=model_cfg.get("max_tokens", 200),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.4),
        max_output_tokens=model_cfg.get("max_output_tokens", 200),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

parser = StrOutputParser()


# ==========================================
# 2️⃣ Input Message (Test Cases)
# ==========================================
messages = [
    "My order arrived late and one item was missing.",
    "Can you share your enterprise pricing plans?",
    "I’m getting an authentication error while using your API.",
]


# ==========================================
# 3️⃣ Define Specialized Prompts
# ==========================================
support_prompt = PromptTemplate(
    input_variables=["message"],
    template=(
        "You are a customer support assistant.\n"
        "Respond professionally to this issue and suggest a resolution.\n\nMessage:\n{message}"
    ),
)
support_chain = support_prompt | llm | parser

sales_prompt = PromptTemplate(
    input_variables=["message"],
    template=(
        "You are a sales representative.\n"
        "Respond warmly, providing pricing and feature details.\n\nMessage:\n{message}"
    ),
)
sales_chain = sales_prompt | llm | parser

tech_prompt = PromptTemplate(
    input_variables=["message"],
    template=(
        "You are a technical support engineer.\n"
        "Provide a concise troubleshooting step or explanation.\n\nMessage:\n{message}"
    ),
)
tech_chain = tech_prompt | llm | parser


# ==========================================
# 4️⃣ Router Prompt — to decide destination
# ==========================================
router_prompt = PromptTemplate(
    input_variables=["message"],
    template=(
        "Classify this message into one of these categories:\n"
        "1. Support\n2. Sales\n3. Technical\n\n"
        "Message:\n{message}\n\n"
        "Only return the category name."
    ),
)
router_chain = router_prompt | llm | parser


# ==========================================
# 5️⃣ Router Logic — RunnableBranch
# ==========================================
router = RunnableBranch(
    # Conditional branches
    (
        lambda x: "support" in x.lower(),
        support_chain,
    ),
    (
        lambda x: "sales" in x.lower() or "pricing" in x.lower(),
        sales_chain,
    ),
    (
        lambda x: "technical" in x.lower() or "api" in x.lower() or "error" in x.lower(),
        tech_chain,
    ),
    # Default fallback
    RunnablePassthrough(),
    # acts as a "do nothing" or "no-op" branch, ensuring the router always returns something, even if no specific condition is met.
)


# ==========================================
# 6️⃣ Complete Workflow — Router → Branch → Response
# ==========================================
def process_message(msg: str):
    """Route message to the correct chain."""
    print("💬 Incoming Message:")
    print(msg)
    print("-----------------------------")

    category = router_chain.invoke({"message": msg})
    response = router.invoke(category)

    print(f"📦 Routed Category: {category}")
    print(f"🧠 Assistant Response:\n{response}\n")
    print("==============================================\n")


# ==========================================
# 7️⃣ Run Demo
# ==========================================
print("🔗 ROUTER CHAIN DEMONSTRATION — Intelligent Message Routing\n")

for msg in messages:
    process_message(msg)

print("✅ Router chain executed successfully!\n")

print("📘 PROCESS OVERVIEW")
print("""
Router Chain Flow:
  - Step 1 → Classify message intent.
  - Step 2 → Route to appropriate domain-specific chain.
  - Step 3 → Generate relevant, contextual reply.

✅ Ideal for:
   - Helpdesk automation
   - CRM assistants
   - Chat routing for enterprise workflows
""")
