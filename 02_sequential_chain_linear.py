"""
02_sequential_chain_linear.py
-----------------------------
Concept: Linear Sequential Chain — realistic workflow.

Use Case: Process a customer complaint:
  1️⃣ Summarize → 2️⃣ Analyze Sentiment → 3️⃣ Rewrite professionally.
"""

import os, json
from dotenv import load_dotenv

# must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# Setup
# ==========================================
load_dotenv()
with open("config.json") as f:
    config = json.load(f)

provider = config["provider"]
cfg = config[provider]

if provider == "openai":
    llm = ChatOpenAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.5),
        max_tokens=cfg.get("max_tokens", 250),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.5),
        max_output_tokens=cfg.get("max_output_tokens", 250),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

parser = StrOutputParser()

# ==========================================
# Input
# ==========================================
customer_complaint = """
I ordered a smartphone last week, and it arrived late by three days.
The battery drains super fast, and the charging cable stopped working on day two.
Customer support didn’t respond to my email yet. This is really frustrating!
"""

# ==========================================
# Step 1 → Summarize
# ==========================================
summarize_prompt = PromptTemplate(
    input_variables=["complaint"],
    template="Summarize this customer complaint in 2 sentences:\n\n{complaint}",
)
summarize_chain = summarize_prompt | llm | parser

# ==========================================
# Step 2 → Sentiment
# ==========================================
sentiment_prompt = PromptTemplate(
    input_variables=["summary"],
    template=(
        "Classify sentiment of this summary as Positive, Neutral, or Negative.\n"
        "Only output one word.\n\nSummary:\n{summary}"
    ),
)
sentiment_chain = sentiment_prompt | llm | parser

# ==========================================
# Step 3 → Rewrite professionally
# ==========================================
rewrite_prompt = PromptTemplate(
    input_variables=["summary", "sentiment"],
    template=(
        "Rewrite the following complaint summary in a polite, empathetic, professional tone.\n"
        "Add sentiment tag at the end in brackets.\n\nSummary:\n{summary}\nSentiment: {sentiment}"
    ),
)
rewrite_chain = rewrite_prompt | llm | parser


# ==========================================
# Linear Pipeline (Functional Composition)
# ==========================================
def process_complaint(complaint: str) -> str:
    summary = summarize_chain.invoke({"complaint": complaint})
    sentiment = sentiment_chain.invoke({"summary": summary})
    rewritten = rewrite_chain.invoke({"summary": summary, "sentiment": sentiment})
    return rewritten


# ==========================================
# Run Demo
# ==========================================
print("🔗 Sequential Chain — Customer Complaint Workflow\n")
result = process_complaint(customer_complaint)

print("📋 Final Processed Output:\n")
print(result)
print("\n✅ Sequential chain executed successfully!\n")

print("📘 PROCESS OVERVIEW")
print("""
1️⃣  Summarize complaint → short, concise.
2️⃣  Analyze sentiment → classify tone.
3️⃣  Rewrite → professional, empathetic format.

✅ Practical Uses:
   - Customer support ticket triage
   - Feedback dashboards
   - Complaint auto-summarization
""")
