"""
08_chains_summary_demo.py
--------------------------
📘 LangChain “Chains” Recap — All Core Patterns in One File

Demonstrates:
  1️⃣ No Chain (manual prompt)
  2️⃣ Simple Runnable Chain (prompt | llm)
  3️⃣ Sequential Chain (linear)
  4️⃣ Non-linear (multi-branch) Chain
  5️⃣ Router Chain (conditional branching)
  6️⃣ Hybrid Chain (rule + LLM + escalation)
  7️⃣ Pre-Post Processing + Cache

Each mini-demo is short, functional, and mirrors a real-world use case.

Perfect as a final lecture summary.
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
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

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
# 2️⃣ No Chain (manual LLM call)
# ==========================================
print("\n=== 1️⃣ NO CHAIN — Manual LLM Call ===")
prompt_text = "Summarize this: 'LangChain helps developers build AI workflows quickly and modularly.'"
response = llm.invoke(prompt_text)
print(response.content.strip())


# ==========================================
# 3️⃣ Simple Runnable Chain
# ==========================================
print("\n=== 2️⃣ SIMPLE RUNNABLE CHAIN (prompt | llm) ===")
simple_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in one concise paragraph for a beginner."
)
chain = simple_prompt | llm | parser
print(chain.invoke({"topic": "Machine Learning"}))


# ==========================================
# 4️⃣ Sequential (Linear) Chain
# ==========================================
print("\n=== 3️⃣ SEQUENTIAL CHAIN (Linear) ===")
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize this complaint: {text}",
)
sentiment_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Classify sentiment (Positive/Neutral/Negative): {summary}",
)

summary_chain = summary_prompt | llm | parser
sentiment_chain = sentiment_prompt | llm | parser

text = "My package arrived two days late and the box was slightly damaged."
summary = summary_chain.invoke({"text": text})
sentiment = sentiment_chain.invoke({"summary": summary})
print(f"Summary → {summary}\nSentiment → {sentiment}")


# ==========================================
# 5️⃣ Non-Linear (Multi-Branch) Chain
# ==========================================
print("\n=== 4️⃣ NON-LINEAR CHAIN (Multi-Branch) ===")
desc = "The EcoBrew Smart Kettle supports Wi-Fi, app control, and auto shutoff for energy saving."

features_prompt = PromptTemplate(
    input_variables=["desc"],
    template="List 3 key features of: {desc}",
)
seo_prompt = PromptTemplate(
    input_variables=["desc"],
    template="Generate 5 SEO keywords for this product: {desc}",
)
ad_prompt = PromptTemplate(
    input_variables=["desc"],
    template="Write a 2-sentence product ad: {desc}",
)

multi_chain = RunnableParallel({
    "features": features_prompt | llm | parser,
    "seo": seo_prompt | llm | parser,
    "ad": ad_prompt | llm | parser,
    "input": RunnablePassthrough(),
})
multi_result = multi_chain.invoke({"desc": desc})
print(json.dumps(multi_result, indent=2))


# ==========================================
# 6️⃣ Router Chain (Conditional Branching)
# ==========================================
print("\n=== 5️⃣ ROUTER CHAIN ===")
msg = "Can you share your pricing details?"

router_prompt = PromptTemplate(
    input_variables=["msg"],
    template="Classify this message as Support, Sales, or Technical: {msg}",
)
category = (router_prompt | llm | parser).invoke({"msg": msg})

if "support" in category.lower():
    reply = f"[Support Reply] Sorry for the issue! Our team will check your case immediately."
elif "sales" in category.lower():
    reply = f"[Sales Reply] Our pricing starts at $299/month with premium features."
else:
    reply = f"[Tech Reply] Please check your API key or request headers."
print(f"Message → {msg}\nCategory → {category}\nReply → {reply}")


# ==========================================
# 7️⃣ Hybrid Chain (Rule + LLM)
# ==========================================
print("\n=== 6️⃣ HYBRID CHAIN (Rule + LLM) ===")
complaint = "I was charged twice and haven’t received a refund in 2 weeks!"

summarize = (PromptTemplate(
    input_variables=["complaint"],
    template="Summarize this complaint briefly: {complaint}"
) | llm | parser).invoke({"complaint": complaint})

severity = "high" if "refund" in summarize.lower() or "charged" in summarize.lower() else "low"

reply_template = PromptTemplate(
    input_variables=["summary", "severity"],
    template="Write a professional reply for a {severity} severity complaint:\n{summary}",
)
reply = (reply_template | llm | parser).invoke({"summary": summarize, "severity": severity})
print(f"Summary → {summarize}\nSeverity → {severity}\nReply → {reply}")


# ==========================================
# 8️⃣ Pre/Post + Cache
# ==========================================
print("\n=== 7️⃣ PRE + POST PROCESSING + CACHE ===")
cache = {}

def preprocess(text: str):
    return text.lower().strip().replace("pls", "please")

def postprocess(resp: str, original: str):
    return json.dumps({
        "original": original,
        "cleaned": preprocess(original),
        "response": resp.strip(),
    }, indent=2)

email = "Hey, I orderd a phone last week but not yet arrived. pls help!"
if email in cache:
    print("⚡ Using cached result!")
else:
    cleaned = preprocess(email)
    resp = (PromptTemplate(
        input_variables=["msg"],
        template="Write a short, polite reply for this message:\n{msg}"
    ) | llm | parser).invoke({"msg": cleaned})
    cache[email] = postprocess(resp, email)
print(cache[email])


print("\n✅ SUMMARY DEMO COMPLETED SUCCESSFULLY!")
print("""
📘 Concepts Recap:
1️⃣ No Chain — Direct LLM call
2️⃣ Simple Chain — prompt | llm
3️⃣ Sequential — step-by-step pipeline
4️⃣ Non-linear — multiple parallel branches
5️⃣ Router — dynamic message routing
6️⃣ Hybrid — combine rules + AI
7️⃣ Pre/Post + Cache — real-world polish

✅ Learners now understand the full Chain spectrum in LangChain!
""")
