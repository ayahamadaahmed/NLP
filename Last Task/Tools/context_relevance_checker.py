import json
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def build_context_relevance_tool(llm):

    template = """
Determine if the provided context is relevant to the user's question.

Return ONLY one of:
- relevant
- irrelevant
- unclear

Context:
{context}

Question:
{question}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    def _run(input_str: str):
        # input comes as a STRING from the agent, not dict
        try:
            data = json.loads(input_str.replace("'", "\""))
        except:
            return "unclear"

        context = data.get("context", "")
        question = data.get("question", "")

        result = chain.run(context=context, question=question).strip().lower()

        if "relevant" in result:
            return "relevant"
        if "irrelevant" in result:
            return "irrelevant"
        return "unclear"

    return Tool(
        name="ContextRelevanceChecker",
        func=_run,
        description="Checks whether context is relevant to the question. Expects a JSON string."
    )
