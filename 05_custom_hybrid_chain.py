"""
05_custom_hybrid_chain.py
--------------------------
Concept: Demonstrate a hybrid (LLM + rule-based + human-in-loop) custom chain.
Use Case: Customer complaint triage & escalation system.

Workflow:
  1️⃣ LLM summarizes the complaint.
  2️⃣ Rule-based logic evaluates severity.
  3️⃣ If severe → escalate to human.
     Else → LLM drafts polite auto-reply.

LangChain Concept: Custom Runnable + conditional branching.
"""

import os
import json
from dotenv import load_dotenv

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ==========================================
# 1️⃣ Setup
# ==========================================
load_dotenv()

with open("config.json", "r") as f:
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
# 2️⃣ Input: Simulated customer complaints
# ==========================================
complaints = [
    "My delivery was delayed by a day, but I finally received the product.",
    "I’ve been charged twice for my order and still haven’t received a refund after two weeks!",
    "The product arrived completely broken. No response from support even after 5 emails!",
]


# ==========================================
# 3️⃣ Step 1 — LLM Summarization Chain
# ==========================================
summary_prompt = PromptTemplate(
    input_variables=["complaint"],
    template="Summarize this customer complaint briefly:\n\n{complaint}",
)
summarize_chain = summary_prompt | llm | parser


# ==========================================
# 4️⃣ Step 2 — Rule-based severity evaluator
# ==========================================
def evaluate_severity(summary: str) -> str:
    """Simple heuristic-based severity detection."""
    summary_lower = summary.lower()

    # Severity scoring
    severe_keywords = ["refund", "broken", "charged", "fraud", "not received", "angry"]
    moderate_keywords = ["late", "delay", "slow", "damaged packaging"]

    if any(word in summary_lower for word in severe_keywords):
        return "high"
    elif any(word in summary_lower for word in moderate_keywords):
        return "medium"
    return "low"


# ==========================================
# 5️⃣ Step 3 — LLM for auto-reply generation
# ==========================================
reply_prompt = PromptTemplate(
    input_variables=["summary", "severity"],
    template=(
        "You are a polite customer support assistant.\n"
        "Given the complaint summary below, write a short professional reply.\n"
        "If severity is 'high', express empathy and mention escalation.\n"
        "If severity is 'medium' or 'low', acknowledge and suggest resolution steps.\n\n"
        "Complaint Summary: {summary}\nSeverity Level: {severity}\n\nReply:"
    ),
)
reply_chain = reply_prompt | llm | parser


# ==========================================
# 6️⃣ Full Hybrid Chain
# ==========================================
def process_complaint(complaint: str) -> None:
    print("💬 Incoming Complaint:")
    print(complaint)
    print("--------------------------------------------")

    # Step 1: Summarize complaint using LLM
    summary = summarize_chain.invoke({"complaint": complaint})
    print(f"📝 Summary: {summary}")

    # Step 2: Evaluate severity (rule-based)
    severity = evaluate_severity(summary)
    print(f"⚙️ Detected Severity: {severity.upper()}")

    # Step 3: Handle based on severity
    if severity == "high":
        print("🚨 Escalation Required: Forwarding to human supervisor.")
        response = reply_chain.invoke({"summary": summary, "severity": severity})
        print(f"🤖 Auto-Drafted Message (for Supervisor):\n{response}\n")
    else:
        response = reply_chain.invoke({"summary": summary, "severity": severity})
        print(f"🤖 Auto-Reply to Customer:\n{response}\n")

    print("============================================\n")


# ==========================================
# 7️⃣ Run the Demo
# ==========================================
print("🔗 HYBRID CHAIN — Automated + Human-in-Loop Escalation\n")

for c in complaints:
    process_complaint(c)

print("✅ Hybrid Chain executed successfully!\n")

print("📘 PROCESS OVERVIEW")
print("""
Hybrid Chain combines:
  🤖 LLM reasoning (summarization + reply generation)
  ⚙️ Rule-based logic (severity classification)
  👤 Human supervision (for high-severity cases)

✅ Ideal for:
   - Customer escalation systems
   - Feedback triage
   - Compliance-aware automation pipelines
""")
