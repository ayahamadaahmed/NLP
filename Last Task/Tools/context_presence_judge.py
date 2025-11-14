import os
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def build_context_presence_tool(llm):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROMPT_PATH = os.path.join(BASE_DIR, "..", "prompts", "context_judge_prompt.txt")

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template_text = f.read()

    prompt = PromptTemplate.from_template(template_text)

    chain = LLMChain(llm=llm, prompt=prompt)

    def _run(input_text):
        result = chain.run(input=input_text).strip().lower()
        if "context_provided" in result:
            return "context_provided"
        return "context_missing"

    return Tool(
        name="ContextPresenceJudge",
        func=_run,
        description="Checks whether the user provided context."
    )
