from govee_assistant.govee_client import GoveeClient, GoveeAPIError

def main():
    client = GoveeClient()
    devices = client.list_devices()
    print(f"Found {len(devices)} device(s):\n")

    for d in devices:
        print(f"- {d.device_name}  (sku={d.sku}, id={d.device_id}, type={d.device_type})")
        cap_names = [f"{c.type.split('.')[-1]}.{c.instance}" for c in d.capabilities]
        print(f"  capabilities: {cap_names}")
        try:
            state = client.get_state(d.sku, d.device_id)
            print(f"  state: {state}")
        except GoveeAPIError as e:
            print(f"  state: <error: {e}>")
        print()


if __name__ == "__main__":
    main()
