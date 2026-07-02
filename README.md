# Creative Fictional News Agency

This project adapts the LangGraph research-assistant pattern into a fictional newsletter studio. A small graph now coordinates four steps:

- News collector gathers inspiration from recent headlines.
- Fiction writers turn each headline into a clearly fictional article.
- A creative editor merges those drafts into a single hybrid story.
- A fact-checker and creativity evaluator reviews the final piece before it is saved.

## Run it

```bash
python research_assistant.py
```

The generated newsletter will be written to outputs/daily_newsletter.md.

If you have API keys configured for OpenAI or Google Generative AI, the workflow can use them. Otherwise it will fall back to a built-in draft so you can still run the pipeline locally.