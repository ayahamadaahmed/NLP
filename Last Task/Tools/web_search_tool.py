from langchain.tools import Tool

def build_web_search_tool():

    def _fake_search(query: str):
        # Fake result to avoid API errors
        return f"(FAKE SEARCH RESULT) Query: {query}"

    return Tool(
        name="WebSearchTool",
        func=_fake_search,
        description="Simulated web search that returns mock results. Does NOT require an API key."
    )
