# python -m tests.test_app_build

from app import build_ui
from .test_tools_offline import FakeGoveeClient, build_devices
class DummyBackend:
    backend_name = "cpu (stub)"

class DummyAgent:
    def __init__(self):
        self.backend = DummyBackend()

    def chat(self, message, history):
        new_history = (history or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"stub reply to: {message}"},
        ]
        return f"stub reply to: {message}", new_history

def main():
    client = FakeGoveeClient(build_devices())
    agent = DummyAgent()

    demo = build_ui(client, agent)
    assert demo is not None

    devices = client.list_devices()
    print(f"Built UI OK with {len(devices)} devices:")
    for d in devices:
        controls = []
        if d.has("devices.capabilities.on_off", "powerSwitch"):
            controls.append("power")
        if d.has("devices.capabilities.range", "brightness"):
            controls.append("brightness")
        print(f"  - {d.device_name}: {controls or 'read-only'}")


if __name__ == "__main__":
    main()
