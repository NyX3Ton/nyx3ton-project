import logging

from govee_assistant import config
from govee_assistant.agent import GoveeAgent
from govee_assistant.govee_client import GoveeClient

logging.basicConfig(level=logging.INFO)


def _build_agent(client):
    if config.GOVEE_AGENT_MODE == "workflow":
        from govee_assistant.orchestrator import OrchestratedAgent
        agent = OrchestratedAgent(client)
    else:
        agent = GoveeAgent(client)
    if config.GOVEE_CRITIQUE_ENABLED:
        from govee_assistant.agent import WriterCriticAgent
        agent = WriterCriticAgent(agent)
    return agent


def main():
    client = GoveeClient()
    agent = _build_agent(client)
    print(f"\nAgent ready (mode={config.GOVEE_AGENT_MODE}, backend={agent.backend.backend_name}). Type 'quit' to exit.\n")

    history: list[dict] = []
    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() in {"quit", "exit"}:
            break
        reply, history = agent.chat(user_msg, history)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
