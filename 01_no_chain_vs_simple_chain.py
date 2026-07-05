"""
01_no_chain_vs_simple_chain.py (Updated for LangChain v0.2+)
------------------------------------------------------------
Concept: Demonstrate the difference between manual LLM calls and using a Runnable chain.

Use Case: Summarizing customer feedback from an e-commerce platform.

LangChain Concept: RunnableSequence (prompt | llm)

This is the modern replacement for the deprecated LLMChain class.
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


# ==========================================
# 1️⃣ Setup & Model Loading
# ==========================================
load_dotenv()

with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
model_cfg = config[provider]

if provider == "openai":
    llm = ChatOpenAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.5),
        max_tokens=model_cfg.get("max_tokens", 200),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif provider == "gemini":
    llm = ChatGoogleGenerativeAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.5),
        max_output_tokens=model_cfg.get("max_output_tokens", 200),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
else:
    raise ValueError("Invalid provider in config.json — must be 'openai' or 'gemini'.")


# ==========================================
# 2️⃣ Use Case: Summarizing customer feedback
# ==========================================
customer_feedback = """
I ordered a noise-cancelling headphone last week. 
Delivery was super fast, but the packaging was slightly damaged.
The sound quality is fantastic, and battery life easily lasts 2 days.
However, the Bluetooth sometimes disconnects when I move around.
"""

# ==========================================
# 3️⃣ Manual LLM Call (No Chain)
# ==========================================
manual_prompt = f"""
Summarize the following customer feedback in 2-3 sentences.
Highlight positives and negatives clearly.

Feedback:
{customer_feedback}
"""

print("🧾 MANUAL LLM CALL (Without Chain)")
print("----------------------------------")
manual_response = llm.invoke(manual_prompt)
print(manual_response.content.strip())
print("\n----------------------------------\n")


# ==========================================
# 4️⃣ Runnable Chain (Modern Replacement for LLMChain)
# ==========================================
print("🔗 RUNNABLE CHAIN IMPLEMENTATION (With prompt | llm)")
print("---------------------------------------------------")

prompt_template = PromptTemplate(
    input_variables=["feedback"],
    template=(
        "Summarize the following customer feedback in 2-3 sentences.\n"
        "Highlight both the positives and negatives clearly.\n\n"
        "Feedback:\n{feedback}"
    ),
)

# Modern chaining using RunnableSequence (| operator)
chain = prompt_template | llm

# Run the chain
response = chain.invoke({"feedback": customer_feedback})
print(response.content.strip())
print("\n✅ Runnable Chain executed successfully!\n")


# ==========================================
# 5️⃣ Comparison & Takeaway
# ==========================================
print("📘 COMPARISON SUMMARY")
print("----------------------")
print("""
1. Manual Approach:
   - Prompt string created manually.
   - Direct LLM invocation each time.
   - Harder to reuse, maintain, or scale.

2. Runnable Chain (Modern LangChain):
   - Uses prompt | llm for clean, functional composition.
   - No deprecated classes (LLMChain).
   - Seamlessly extendable into multi-step workflows.
""")
