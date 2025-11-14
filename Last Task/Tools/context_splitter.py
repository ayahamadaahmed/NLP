from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def build_context_splitter_tool(llm):

    template = """
Separate the user input into two parts:
1. background_context
2. actual_question

User Input:
{input}

Return JSON ONLY in this format:
{{
  "context": "...",
  "question": "..."
}}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["input"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    def _run(user_input):
        result = chain.run(input=user_input)
        return result
    
    return Tool(
        name="ContextSplitter",
        func=_run,
        description="Separates background context from the actual question. Returns JSON with context and question."
    )
