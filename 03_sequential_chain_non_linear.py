"""
03_sequential_chain_non_linear.py
---------------------------------
Concept: Demonstrate a Non-Linear (Multi-Branch) Sequential Chain.

Use Case: Product intelligence automation system.
  - Input: Product description
  - Branches:
      1️⃣ Extract key features
      2️⃣ Generate ad copy
      3️⃣ Generate SEO tags
  - Merge: Structured summary output

LangChain Concept: RunnableParallel + RunnablePassthrough
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
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# ==========================================
# 1️⃣ Setup & Model
# ==========================================
load_dotenv()

with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
model_cfg = config[provider]

if provider == "openai":
    llm = ChatOpenAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.6),
        max_tokens=model_cfg.get("max_tokens", 250),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.6),
        max_output_tokens=model_cfg.get("max_output_tokens", 250),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

parser = StrOutputParser()


# ==========================================
# 2️⃣ Input
# ==========================================
product_description = """
Introducing the new EcoBrew Smart Coffee Maker — an energy-efficient device
that brews coffee via a smartphone app. It supports WiFi connectivity,
custom brew settings, and eco-friendly pods made from biodegradable materials.
Ideal for modern homes and sustainable coffee lovers.
"""


# ==========================================
# 3️⃣ Step 1: Extract Product Features
# ==========================================
feature_prompt = PromptTemplate(
    input_variables=["description"],
    template=(
        "Extract the 5 key product features from the following description:\n\n"
        "{description}\n\nReturn as a bullet list."
    ),
)
feature_chain = feature_prompt | llm | parser


# ==========================================
# 4️⃣ Step 2: Generate Ad Copy
# ==========================================
ad_prompt = PromptTemplate(
    input_variables=["description"],
    template=(
        "Write a short, catchy advertisement for this product (max 3 sentences). "
        "Highlight its unique qualities:\n\n{description}"
    ),
)
ad_chain = ad_prompt | llm | parser


# ==========================================
# 5️⃣ Step 3: Generate SEO Tags
# ==========================================
seo_prompt = PromptTemplate(
    input_variables=["description"],
    template=(
        "Generate 10 SEO-friendly keywords or hashtags for this product. "
        "Return as a comma-separated list.\n\n{description}"
    ),
)
seo_chain = seo_prompt | llm | parser


# ==========================================
# 6️⃣ Combine Outputs (RunnableParallel)
# ==========================================
multi_chain = RunnableParallel(
    {
        "features": feature_chain,
        "ad_copy": ad_chain,
        "seo_tags": seo_chain,
        "original_input": RunnablePassthrough(), #rint(printspecial LangChain utility that simply "passes through" the input it receives, without any processing.
    }
)


# ==========================================
# 7️⃣ Post-Processing (Combine)
# ==========================================
def combine_results(results: dict) -> str:
    """Combine all outputs into a structured final summary."""
    final_report = f"""
📦 PRODUCT INTELLIGENCE REPORT
-------------------------------
📝 Original Input:
{results['original_input']}

🌟 Key Features:
{results['features']}

💬 Ad Copy:
{results['ad_copy']}

🏷️ SEO Keywords:
{results['seo_tags']}
"""
    return final_report.strip()


# ==========================================
# 8️⃣ Execute Full Workflow
# ==========================================
print("🔗 Non-Linear Sequential Chain — Product Intelligence System\n")

results = multi_chain.invoke({"description": product_description})
final_output = combine_results(results)

print(final_output)
print("\n✅ Non-linear sequential chain executed successfully!\n")

print("📘 PROCESS OVERVIEW")
print("""
Parallel branches:
  - Extracts structured product features
  - Generates creative ad copy
  - Produces SEO keywords
Then merges results into a unified marketing intelligence report.
""")
