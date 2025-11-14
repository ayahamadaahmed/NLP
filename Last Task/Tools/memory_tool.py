from langchain.tools import Tool

# simple in-memory store
MEMORY_DB = {
    "history": []
}

def memory_recall(_=None):
    """Return past messages as a single string"""
    if not MEMORY_DB["history"]:
        return "no_memory"
    return "\n".join(MEMORY_DB["history"])

def memory_add(msg: str):
    """Save message into memory"""
    MEMORY_DB["history"].append(msg)
    return "stored"

def build_memory_tool():
    return [
        Tool(
            name="MemoryRecall",
            func=memory_recall,
            description="Retrieve past conversation messages."
        ),
        Tool(
            name="MemoryAdd",
            func=memory_add,
            description="Store important message into memory (User or Assistant at the FINAL answer only)."
        )
    ]
