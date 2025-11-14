# agent/agent_runner.py

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
import os

from tools.context_presence_judge import build_context_presence_tool
from tools.web_search_tool import build_web_search_tool
from tools.context_relevance_checker import build_context_relevance_tool
from tools.context_splitter import build_context_splitter_tool
from tools.memory_tool import build_memory_tool
from tools.context_retriever import build_retriever_tools


def build_agent():
    # -------- LLM --------
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")   # تأكدي إنه موجود في الـ .env
    )

    # -------- Tools --------
    context_tool = build_context_presence_tool(llm)
    web_tool = build_web_search_tool()
    relevance_tool = build_context_relevance_tool(llm)
    splitter_tool = build_context_splitter_tool(llm)

    # Memory tools expected to return [MemoryAdd, MemoryRecall]
    memory_tools = build_memory_tool()

    # RAG tools: [EmbedDocument, RetrieveDocument]
    retriever_tools = build_retriever_tools()

    tools = [
        context_tool,
        web_tool,
        relevance_tool,
        splitter_tool,
        *memory_tools,
        *retriever_tools,
    ]

    # -------- Prompt Template (VERY IMPORTANT) --------
    template = """
You are a context-aware assistant that uses tools via the ReAct pattern.

IMPORTANT CALL FORMAT:
- NEVER call tools like ToolName(...).
- ALWAYS use EXACTLY this format:

  Action: ToolName
  Action Input: "some text"

(no JSON, no extra keys)

TOOLS YOU CAN USE:
{tools}

Typical tool names you will see:
- ContextPresenceJudge        → decide if user message has background context
- WebSearchTool               → search the web ONLY when info is not in memory or PDF
- ContextRelevanceChecker     → check if context is relevant to question
- ContextSplitter             → split a mixed input into {{context, question}}
- MemoryAdd                   → append a short memory string AFTER Final Answer (only then)
- MemoryRecall                → recall recent conversation context
- EmbedDocument               → embed a PDF file given its file path
- RetrieveDocument            → retrieve relevant text from the embedded PDF

RAG RULES (PDF QUESTIONS):
- A PDF is considered "loaded" if:
  - The user has recently used "upload ..." OR
  - You or the tools have successfully called EmbedDocument, OR
  - Previous answers clearly referenced content that came from a PDF.

- IF the question refers to ANY of the following, you MUST CALL RetrieveDocument FIRST:
  - "the pdf", "pdf", "uploaded document", "the document", "the file",
  - "based on the document", "based on the pdf", "from the pdf", "from the file",
  - any follow-up question right after you answered using the PDF,
  - questions like:
      * "what is the project based on the uploaded document?"
      * "what is the content of the pdf?"
      * "what technologies are used?"
      * "what are the objectives?" (after a project PDF)
      * "summarize the pdf", "what is intro", "what is section X"

- When a user writes something like: 
    "upload D:/path/to/file.pdf"
  you should:
    1) Call EmbedDocument with the RAW FILE PATH as Action Input.
       Example:
         Action: EmbedDocument
         Action Input: "D:/path/to/file.pdf"

    2) If the embed succeeds, you are now allowed to use RetrieveDocument
       to answer follow-up questions about that PDF.

- DO NOT use WebSearchTool for questions that can be answered from the uploaded PDF.
- WebSearchTool is ONLY allowed when:
    - RetrieveDocument returns a clear NO_RELEVANT_DOCS signal
    - OR no PDF has ever been embedded
    - OR the user explicitly says: "search the web", "ignore the pdf", etc.

- If RetrieveDocument returns something like "TOO_GENERAL_QUERY":
    - DO NOT call RetrieveDocument again in a loop.
    - Instead reply: 
      "Your question about the PDF is too general. Please ask about a specific topic (e.g., objectives, architecture, technologies, etc.)."

MEMORY RULES:
- MemoryAdd and MemoryRecall are SPECIAL tools.
- During tool reasoning (Thought/Action/Observation), do NOT spam MemoryAdd.
- ONLY call MemoryAdd ONCE after you have produced a Final Answer if there is something worth remembering
  (e.g., user preferences, important long-term facts, that a PDF was uploaded).
- Example after answering a PDF question:
    Action: MemoryAdd
    Action Input: "last_pdf_uploaded=true; file=Ai Contact Center For Hospitals — Graduation Project Proposal"

- Use MemoryRecall when:
    - The user refers to "previous question", "as I said before", "the project we mentioned", etc.
    - You need to re-load short-term context that you might have forgotten.

GENERAL REACT PATTERN:

Question: {input}
Thought: think step-by-step about what to do next
Action: one of [{tool_names}]
Action Input: "text"
Observation: tool result
... (you may repeat Thought → Action → Observation multiple times) ...
Thought: I now know the answer
Final Answer: your answer to the user (in natural language)

VERY IMPORTANT:
- Respect the Action / Action Input / Observation / Final Answer format.
- Do NOT invent new fields.
- Prefer using RetrieveDocument over WebSearchTool for questions that are obviously about the PDF.

Begin!

Question: {input}
{agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template)

    react_agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    return executor
