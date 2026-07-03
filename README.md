# Creative Fictional News Agency

This project adapts the [LangGraph research-assistant example](https://github.com/langchain-ai/langchain-academy/blob/main/module-4/studio/research_assistant.py) into a fictional newsletter studio. A small graph now coordinates four steps:

- News collector gathers inspiration from recent headlines.
- Fiction writers turn each headline into a clearly fictional article.
- A creative editor merges those drafts into a single hybrid story.
- A fact-checker and creativity evaluator reviews the final piece before it is saved.

## Installation

```
python3 -m pip install -r requirements.txt
```

## Run

Make sure your ```TAVILY_API_KEY``` environment variable is set.

```bash
python3 research_assistant.py
```
with the ```-h``` flag to see available run-time options.

If you have API keys configured for OpenAI or Google Generative AI, the workflow can use them. By default it uses a local LLM model (llama3.2).

## Substack

You can subscribe [here](https://liyiwei.substack.com/?utm_campaign=email-home&r=52pf32) to receive the generated stories as newsletters.