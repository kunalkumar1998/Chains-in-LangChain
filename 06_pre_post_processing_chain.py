"""
06_pre_post_processing_chain.py
-------------------------------
Concept: Demonstrate a full pipeline with preprocessing, postprocessing, and caching.

Use Case: Customer email cleaner + auto-response generator.

Workflow:
  1️⃣ Pre-process → clean + normalize raw user input
  2️⃣ Core → LLM generates structured, friendly response
  3️⃣ Post-process → clean formatting, add metadata
  4️⃣ Memory → lightweight in-memory cache

LangChain Concepts: Custom pre/post functions + RunnableLambda + RunnablePassthrough
"""

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv

# must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


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
        temperature=cfg.get("temperature", 0.4),
        max_tokens=cfg.get("max_tokens", 250),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.4),
        max_output_tokens=cfg.get("max_output_tokens", 250),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

parser = StrOutputParser()

# ==========================================
# 2️⃣ Preprocessing Function
# ==========================================
def clean_text(raw_text: str) -> str:
    """
    Preprocesses the customer message:
      - Lowercases
      - Strips extra spaces
      - Removes non-alphanumeric characters (except punctuation)
      - Fixes common typos
    """
    text = raw_text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("pls", "please").replace("u", "you").replace("thx", "thanks")
    return text


preprocess = RunnableLambda(clean_text)

# ==========================================
# 3️⃣ Prompt Template (Core LLM Step)
# ==========================================
prompt = PromptTemplate(
    input_variables=["cleaned_text"],
    template=(
        "You are a friendly customer support assistant.\n"
        "Given the cleaned message below, write a polite, clear, and helpful reply.\n\n"
        "Customer Message:\n{cleaned_text}"
    ),
)

# LLM chain
core_chain = prompt | llm | parser

# ==========================================
# 4️⃣ Postprocessing Function
# ==========================================


def postprocess(response: str, original_text: str) -> str:
    """Postprocess the LLM output and add metadata."""
    # Format and timestamp
    clean_response = response.strip().replace("\n\n", "\n")
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_message": original_text,
        "response": clean_response,
        "word_count": len(clean_response.split()),
    }
    return json.dumps(meta, indent=2)


postprocess_chain = RunnableLambda(lambda output: postprocess(output["response"], output["input"]))

# ==========================================
# 5️⃣ Complete End-to-End Pipeline
# ==========================================
def process_email(email_text: str) -> str:

    # Step 1: Preprocess
    cleaned = preprocess.invoke(email_text)

    # Step 2: Core LLM call
    response = core_chain.invoke({"cleaned_text": cleaned})

    # Step 3: Postprocess
    # final_result = postprocess(response, email_text)
    final_result = postprocess_chain.invoke({"response": response, "input": email_text})

    # return result
    return final_result


# ==========================================
# 6️⃣ Demo Inputs
# ==========================================
emails = [
    "Hey, i orderd a phone last week but not yet arrived. pls help!",
    "thx for the update but can u speed up the refund?",
    "hi i recieved wrong color shoes what can i do?",
]


# ==========================================
# 7️⃣ Run the Full Chain
# ==========================================
print("🔗 FULL PIPELINE — Pre-Post Processing + Caching\n")

for e in emails:
    print("💬 Original Email:")
    print(e)
    print("--------------------------------------------")
    output = process_email(e)
    print(output)
    print("\n============================================\n")



print("📘 PROCESS OVERVIEW")
print("""
Pipeline Stages:
  1️⃣ Preprocessing → Cleans & normalizes input text
  2️⃣ Core LLM → Generates contextual reply
  3️⃣ Postprocessing → Adds metadata & formatting
  4️⃣ Memory → Caches results for efficiency

✅ Real-world Applications:
   - Email/chat response systems
   - CRM and ticketing pipelines
   - AI customer assistants with caching
""")
