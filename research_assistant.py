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
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
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
    artwork_prompt: str
    image_data: str
    evaluation: str
    newsletter: str


def default_topic() -> str:
    # from Monday to Sunday, cycle through topics
    topics = ["World News", "Business", "Finance", "Technology", "Science", "Health", "Culture"]
    return topics[datetime.now().weekday() % len(topics)]

def get_llm(model_name: str, temperature: float):
    # default to run local models to save tokens
    if "llama" in model_name:
        try:
            return ChatOllama(model=model_name, temperature=temperature)
        except Exception as e:
            return ChatOllama(model=DEFAULT_OLLAMA_MODEL, temperature=temperature)
            print(f"Error occurred while initializing Ollama model: {e}")
    elif "gemini" in model_name and os.environ.get("GOOGLE_API_KEY"):
        try:
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        except Exception as e:
            return ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, temperature=temperature)
            print(f"Error occurred while initializing Gemini model: {e}")
    elif "gpt" in model_name and os.environ.get("OPENAI_API_KEY"):
        try:
            return ChatOpenAI(model=model_name, temperature=temperature)
        except Exception as e:
            return ChatOpenAI(model=DEFAULT_OPENAI_MODEL, temperature=temperature)
            print(f"Error occurred while initializing OpenAI model: {e}")
    else:
        raise RuntimeError(
            "No valid LLM option or API key found. Please choose a proper model and set the corresponding API key in your environment variables."
        )
    return None


def build_llm(state: NewsletterState):
    model = state.get("llm_model", DEFAULT_OLLAMA_MODEL)
    temperature = state.get("llm_temperature", DEFAULT_MODEL_TEMPERATURE)
    llm = get_llm(model_name=model, temperature=temperature)
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

    system_message = f"""You are a fiction writer for a satirical, elegant newspaper. Create one vivid and humorous article that is clearly fictional, but feels plausible and grounded in the inspiration you are given. Keep the tone polished and slightly uncanny but not overtly absurd.
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
                content="You are a creative editor. Merge several fictional articles into one polished hybrid story for a daily newsletter. Keep it imaginative, cohesive, humourous, and clearly fictional."
            ),
            HumanMessage(content=f"Combine these drafts into one newsletter story and come up with a compelling title :\n\n{joined_drafts}"),
        ]
    )
    story = response.content

    return {"hybrid_story": story}


def generate_artwork_prompt(state: NewsletterState):
    llm = state.get("llm")

    response = llm.invoke(
        [
            SystemMessage(
                content="You are an artist and creative prompt engineer. Create a detailed text prompt for generating an image that represents the hybrid story. The prompt should be vivid, descriptive, humorous, and suitable for a text-to-image generator."
            ),
            HumanMessage(content=f"Create an image prompt for this hybrid story:\n\n{state['hybrid_story']}"),
        ]
    )
    artwork_prompt = response.content

    return {"artwork_prompt": artwork_prompt}


def generate_image(state: NewsletterState):
    # Import here to avoid issues when running without image generation dependencies
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from PIL import Image
        import io

        # For now, we'll just return a placeholder - in a real implementation,
        # this would use an actual text-to-image model like Stable Diffusion or DALL-E
        # This is a simplified version that shows the structure
        image_prompt = state.get("artwork_prompt", "")

        # In a real implementation, you would call a text-to-image API here
        # For example:
        # from langchain_stability import StabilityAI
        # image_model = StabilityAI(model="stable-diffusion-xl-1024-v1-0")
        # image = image_model.invoke(image_prompt)

        # Placeholder for actual image generation
        return {"image_data": f"Generated image based on prompt: {image_prompt[:100]}..."}
    except ImportError:
        # If dependencies aren't available, just return a placeholder
        return {"image_data": "Image generation not available (missing dependencies)"}


def evaluate_story(state: NewsletterState):
    llm = state.get("llm")

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a fact checker and creativity evaluator. Review the newsletter story for imaginative flair, humor, and for how convincingly it resembles a plausible report without being factual."
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

### Artist's Prompt
{state['artwork_prompt']}

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
builder.add_node("generate_artwork_prompt", generate_artwork_prompt)
builder.add_node("generate_image", generate_image)
builder.add_node("evaluate_story", evaluate_story)
builder.add_node("finalize_newsletter", finalize_newsletter)

builder.add_edge(START, "build_llm")
builder.add_edge("build_llm", "collect_news")
builder.add_conditional_edges("collect_news", dispatch_writers, ["write_article"])
builder.add_edge("write_article", "synthesize_story")
builder.add_edge("synthesize_story", "generate_artwork_prompt")
builder.add_edge("generate_artwork_prompt", "generate_image")
builder.add_edge("generate_image", "evaluate_story")
builder.add_edge("evaluate_story", "finalize_newsletter")
builder.add_edge("finalize_newsletter", END)

graph = builder.compile()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a fictional newsletter.")
    parser.add_argument("--topic", type=str, default=default_topic(), help="Topic for the newsletter (optional).")
    parser.add_argument("--output", type=str, default="outputs/daily_newsletter.md", help="Output path for the generated newsletter.")
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, help="Model to use (optional).")
    parser.add_argument("--temperature", type=float, default=DEFAULT_MODEL_TEMPERATURE, help="Temperature for the LLM (optional).")
    parser.add_argument("--max-news-items", type=int, default=DEFAULT_MAX_NEWS_ITEMS, help="Maximum number of news items to fetch (optional).")

    args = parser.parse_args()

    initial_state = {
        "llm_model": args.model,
        "llm_temperature": args.temperature,
        "max_news_items": args.max_news_items,
        "llm": None,
        "topic": args.topic,
        "news_items": [],
        "article_drafts": [],
        "hybrid_story": "",
        "artwork_prompt": "",
        "image_data": "",
        "evaluation": "",
        "newsletter": "",
    }
    result = graph.invoke(initial_state)

    output_result = output_newsletter(result, args.output)

    print(f"Newsletter generated and saved to: {output_result['output_path']}")
