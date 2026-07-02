# Creative Fictional News Agency

Inspired by the [LangGraph research assistant example](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239974-lesson-4-research-assistant), build a create a creative fictional news agency that produces a daily newsletter inspired by real events but with a creative fictional twist.
Using the LangGraph architecture, the agency will consist of the following agents:
- *News collector* will search for the news topics and articles via [Tavily](https://www.tavily.com/) or another search API.
- *Fiction writer* will write a fictional article based on the news found by the collector.
- *Creative editor* will collect several articles from multiple fiction writers and synthesize a hybrid story that creatively combines the multiple topics.
- *Fact checker and creativity evaluator* will verify both the creativity and apparent factualness of the final article.

Create a substack news letter to share the output from the AI agents above after I look at the final output.