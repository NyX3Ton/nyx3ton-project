import logging

from dotenv import load_dotenv

from agent import GoveeAgent
from govee_client import GoveeClient

logging.basicConfig(level=logging.INFO)
load_dotenv()


def main():
    client = GoveeClient()
    agent = GoveeAgent(client)
    print(f"\nAgent ready (backend={agent.backend.backend_name}). Type 'quit' to exit.\n")

    history: list[dict] = []
    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() in {"quit", "exit"}:
            break
        reply, history = agent.chat(user_msg, history)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
