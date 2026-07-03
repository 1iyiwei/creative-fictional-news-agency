import operator
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict
import argparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

# constants
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_MODEL_TEMPERATURE = 0.7
DEFAULT_MAX_NEWS_ITEMS = 2

class NewsletterState(TypedDict):
    llm_model: str
    llm_temperature: float
    llm: object
    topic: str
    max_news_items: int
    news_items: Annotated[list, operator.add]
    article_drafts: Annotated[list, operator.add]
    hybrid_story: str
    evaluation: str
    newsletter: str


def get_llm(ollama_model: str, temperature: float):
    # default to run local models to save tokens
    if ollama_model:
        return ChatOllama(model=ollama_model, temperature=temperature)

    if os.environ.get("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
    if os.environ.get("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    else:
        raise RuntimeError(
            "No LLM API key found. Please set either GOOGLE_API_KEY or OPENAI_API_KEY in your environment variables."
        )
    return None


def build_llm(state: NewsletterState):
    model = state.get("llm_model", DEFAULT_OLLAMA_MODEL)
    temperature = state.get("llm_temperature", DEFAULT_MODEL_TEMPERATURE)
    llm = get_llm(ollama_model=model, temperature=temperature)
    return {"llm": llm}


def collect_news(state: NewsletterState):
    max_news_items = state.get("max_news_items", DEFAULT_MAX_NEWS_ITEMS)
    topic = state.get("topic", "")
    llm = state.get("llm")

    if not llm:
        raise RuntimeError("LLM is not initialized. Please ensure the LLM is built before collecting news.")

    if os.environ.get("TAVILY_API_KEY"):
        try:
            tavily_search = TavilySearch(max_results=max_news_items)
            query_string = "recent headline news" + (f" about {topic}" if topic else "")

            raw_results = tavily_search.invoke({"query": query_string})
            search_results = raw_results.get("results", raw_results) if isinstance(raw_results, dict) else raw_results
            items = []
            for result in search_results[:max_news_items]:
                if isinstance(result, dict):
                    items.append(
                        {
                            "title": result.get("title", "Untitled"),
                            "summary": result.get("content", result.get("snippet", "")),
                            "url": result.get("url", ""),
                            "topic": topic,
                        }
                    )
            if items:
                return {"news_items": items}
        except Exception:
            pass
    else:
        raise RuntimeError(
            "Tavily API key not found. Please set TAVILY_API_KEY in your environment variables to fetch news items."
        )


def dispatch_writers(state: NewsletterState):
    news_items = state.get("news_items", [])
    llm = state.get("llm")
    return [
        Send("write_article", {"llm": llm, "news_item": item, "topic": state.get("topic")})
        for item in news_items
    ]


def write_article(state: NewsletterState):
    news_item = state["news_item"]
    llm = state.get("llm")

    if not llm:
        raise RuntimeError("LLM is not initialized. Please ensure the LLM is built before writing articles.")

    system_message = f"""You are a fiction writer for a satirical, elegant newspaper. Create one vivid article that is clearly fictional, but feels plausible and grounded in the inspiration you are given. Keep the tone polished and slightly uncanny.
    """

    response = llm.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Write a fictional article inspired by this news item: {news_item['summary']}"),
        ]
    )
    body = response.content

    article = {
        "title": f"{news_item['title']} Reimagined",
        "body": body,
        "source_title": news_item["title"],
    }
    return {"article_drafts": [article]}


def synthesize_story(state: NewsletterState):
    llm = state.get("llm")

    drafts = state.get("article_drafts", [])
    joined_drafts = "\n\n".join([f"## {draft['title']}\n{draft['body']}" for draft in drafts])

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a creative editor. Merge several fictional articles into one polished hybrid story for a daily newsletter. Keep it imaginative, cohesive, and clearly fictional."
            ),
            HumanMessage(content=f"Combine these drafts into one newsletter story:\n\n{joined_drafts}"),
        ]
    )
    story = response.content

    return {"hybrid_story": story}


def evaluate_story(state: NewsletterState):
    llm = state.get("llm")

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a fact checker and creativity evaluator. Review the newsletter story for imaginative flair and for how convincingly it resembles a plausible report without being factual."
            ),
            HumanMessage(content=f"Evaluate this newsletter story:\n\n{state['hybrid_story']}"),
        ]
    )
    evaluation = response.content

    return {"evaluation": evaluation}


def finalize_newsletter(state: NewsletterState):
    today = datetime.now().strftime("%B %d, %Y")
    topic = state.get("topic", "daily dispatch")
    newsletter = f"""
# The Midnight Ledger
## {today}
### {topic}

{state['hybrid_story']}

### Sources
{', '.join([draft['source_title'] for draft in state.get('article_drafts', [])])}

### Editorial Review
{state['evaluation']}

---
This issue is a fictional newsletter inspired by real events, written to feel plausible without claiming to be factual.
"""

    return {"newsletter": newsletter}

def output_newsletter(state: NewsletterState, output_path: str):
    newsletter = state.get("newsletter", "")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(newsletter, encoding="utf-8")
    return {"output_path": str(output_path)}

builder = StateGraph(NewsletterState)
builder.add_node("build_llm", build_llm)
builder.add_node("collect_news", collect_news)
builder.add_node("write_article", write_article)
builder.add_node("synthesize_story", synthesize_story)
builder.add_node("evaluate_story", evaluate_story)
builder.add_node("finalize_newsletter", finalize_newsletter)

builder.add_edge(START, "build_llm")
builder.add_edge("build_llm", "collect_news")
builder.add_conditional_edges("collect_news", dispatch_writers, ["write_article"])
builder.add_edge("write_article", "synthesize_story")
builder.add_edge("synthesize_story", "evaluate_story")
builder.add_edge("evaluate_story", "finalize_newsletter")
builder.add_edge("finalize_newsletter", END)

graph = builder.compile()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a fictional newsletter.")
    parser.add_argument("--topic", type=str, default="", help="Topic for the newsletter (optional).")
    parser.add_argument("--output", type=str, default="outputs/daily_newsletter.md",
     help="Output path for the generated newsletter.")
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL, help="Ollama model to use (optional).")
    parser.add_argument("--temperature", type=float, default=DEFAULT_MODEL_TEMPERATURE, help="Temperature for the LLM (optional).")
    parser.add_argument("--max-news-items", type=int, default=DEFAULT_MAX_NEWS_ITEMS, help="Maximum number of news items to fetch (optional).")

    args = parser.parse_args()

    initial_state = {
        "llm_model": args.ollama_model,
        "llm_temperature": args.temperature,
        "max_news_items": args.max_news_items,
        "llm": None,
        "topic": args.topic,
        "news_items": [],
        "article_drafts": [],
        "hybrid_story": "",
        "evaluation": "",
        "newsletter": "",
    }
    result = graph.invoke(initial_state)

    output_result = output_newsletter(result, args.output)

    print(f"Newsletter generated and saved to: {output_result['output_path']}")
