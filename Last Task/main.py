# main.py

from agent.agent_runner import build_agent

def main():
    agent = build_agent()
    print("Chat with Your Context - Running...")

    GREET = ["hi", "hello", "hey", "hii", "good morning", "good evening"]

    while True:
        try:
            text = input("You: ")
        except EOFError:
            break

        if not text.strip():
            continue

        if text.lower().strip() == "exit":
            break

        # 1) Greeting short-circuit (no tools)
        if text.lower().strip() in GREET:
            print("Bot: Hello! How can I help you today?")
            continue

        # 2) Let the AGENT decide what to do with 'upload ...'
        #    (you already gave it rules in the prompt to call EmbedDocument)
        response = agent.invoke({"input": text})

        if isinstance(response, dict):
            # Typical AgentExecutor returns {"output": "...", ...}
            print("Bot:", response.get("output", response))
        else:
            print("Bot:", response)


if __name__ == "__main__":
    main()
