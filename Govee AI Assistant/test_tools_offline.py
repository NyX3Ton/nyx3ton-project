import json

import semantic_match
from agent import GoveeTools, TOOL_CALL_RE, DeviceNotFoundError
from govee_client import Capability, Device

class FakeGoveeClient:
    def __init__(self, devices):
        self._devices = devices
        self.calls = []

    def list_devices(self, force_refresh=False):
        return self._devices

    def get_state(self, sku, device_id):
        return {"online": True, "powerSwitch": 1, "brightness": 80}

    def control(self, sku, device_id, cap_type, instance, value):
        self.calls.append((sku, device_id, cap_type, instance, value))
        return {"code": 200, "message": "success"}

    def set_power(self, device, on):
        return self.control(device.sku, device.device_id, "devices.capabilities.on_off", "powerSwitch", 1 if on else 0)

    def set_brightness(self, device, percent):
        cap = device.capability("devices.capabilities.range", "brightness")
        if cap and cap.value_range:
            lo, hi = cap.value_range["min"], cap.value_range["max"]
            percent = max(lo, min(hi, percent))
        return self.control(device.sku, device.device_id, "devices.capabilities.range", "brightness", percent)

    def set_color_rgb(self, device, r, g, b):
        return self.control(
            device.sku, device.device_id, "devices.capabilities.color_setting", "colorRgb", (r << 16) + (g << 8) + b
        )

    def set_color_temp(self, device, kelvin):
        return self.control(device.sku, device.device_id, "devices.capabilities.color_setting", "colorTemperatureK", kelvin)

    def set_scene(self, device, scene_value, instance="lightScene"):
        return self.control(device.sku, device.device_id, "devices.capabilities.dynamic_scene", instance, scene_value)


class FakeEmbeddingModel:
    VOCAB = ["bedroom", "light", "fan", "tower", "thermometer", "wifi"]
    SYNONYMS = {"lamp": "light", "bulb": "light"}

    def encode(self, texts, normalize_embeddings=True):
        import re as _re

        import numpy as np

        vecs = []
        for text in texts:
            words = {self.SYNONYMS.get(w, w) for w in _re.findall(r"[a-z]+", text.lower())}
            vecs.append([1.0 if v in words else 0.0 for v in self.VOCAB])
        arr = np.array(vecs, dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            arr = arr / norms
        return arr


def build_devices():
    bedroom_light_2 = Device(
        sku="H6006", device_id="56:E2:D0:C9:07:14:6C:40", device_name="Bedroom Light 2",
        device_type="devices.types.light",
        capabilities=[
                        Capability("devices.capabilities.on_off", "powerSwitch",
                                                {"dataType": "ENUM", "options": [{"name": "on", "value": 1}, {"name": "off", "value": 0}]}),
                        Capability("devices.capabilities.range", "brightness",
                                                {"dataType": "INTEGER", "range": {"min": 1, "max": 100, "precision": 1}}),
                        Capability("devices.capabilities.color_setting", "colorRgb",
                                                {"dataType": "INTEGER", "range": {"min": 0, "max": 16777215}}),
                        Capability("devices.capabilities.color_setting", "colorTemperatureK",
                                                {"dataType": "INTEGER", "range": {"min": 2000, "max": 9000}}),
                        Capability("devices.capabilities.dynamic_scene", "lightScene",
                                                {"dataType": "ENUM", "options": [{"name": "Sunrise", "value": 3054}, {"name": "Christmas", "value": 3065}]}),
                    ],
                        )
    bedroom_light = Device(
        sku="H6006", device_id="71:D1:D0:C9:07:1C:B0:E8", device_name="Bedroom Light",
        device_type="devices.types.light",
        capabilities=[
                        Capability("devices.capabilities.on_off", "powerSwitch",
                                                {"dataType": "ENUM", "options": [{"name": "on", "value": 1}, {"name": "off", "value": 0}]}),
                        Capability("devices.capabilities.range", "brightness",
                                                {"dataType": "INTEGER", "range": {"min": 1, "max": 100, "precision": 1}})])
    tower_fan = Device(
        sku="H7107", device_id="17:91:5C:E7:53:B6:0E:0C", device_name="Tower Fan",
        device_type="devices.types.fan",
        capabilities=[
                        Capability("devices.capabilities.on_off", "powerSwitch",{"dataType": "ENUM", "options": [{"name": "on", "value": 1}, {"name": "off", "value": 0}]}),
                        Capability("devices.capabilities.toggle", "oscillationToggle",{"dataType": "ENUM", "options": [{"name": "on", "value": 1}, {"name": "off", "value": 0}]}),
                        Capability("devices.capabilities.work_mode", "workMode", {
                                                                "dataType": "STRUCT",
                                                                "fields": [
                                                                            {"fieldName": "workMode", "dataType": "ENUM", "options": [
                                                                                        {"name": "gearMode", "value": 1}, {"name": "Auto", "value": 3},
                                                                                                                                    ]},
                                                                            {"fieldName": "modeValue", "dataType": "ENUM", "options": [
                                                                                        {"name": "gearMode", "options": [
                                                                                            {"name": "Low", "value": 1}, {"name": "Medium", "value": 2}, {"name": "High", "value": 3},
                                                                            ]},
                                                                            {"name": "Auto", "defaultValue": 22},
                                                                            ]}]})])
    thermometer = Device(
        sku="H5179", device_id="4A:CE:EF:2F:40:46:31:70", device_name="Wifi Thermometer",
        device_type="devices.types.thermometer",
        capabilities=[
                        Capability("devices.capabilities.property", "sensorTemperature", {}),
                        Capability("devices.capabilities.property", "sensorHumidity", {}),
                        ],
                    )

    tower_fan_numbered = Device(
        sku="H7107", device_id="AA:AA:AA:AA:AA:AA:AA:AA", device_name="Tower Fan Numbered",
        device_type="devices.types.fan",
        capabilities=[
                        Capability("devices.capabilities.work_mode", "workMode", {
                                                                                    "dataType": "STRUCT",
                                                                                    "fields": [
                                                                                                {"fieldName": "workMode", "dataType": "ENUM", "options": [
                                                                                    {"name": "Custom", "value": 1}, {"name": "Auto", "value": 3},
                                                                                        ]},
                                                                                {"fieldName": "modeValue", "dataType": "ENUM", "options": [
                        {"name": "Custom", "options": [{"value": v} for v in range(1, 9)]},
                        {"name": "Auto", "defaultValue": 0}]}]})])

    tower_fan_ranged = Device(
        sku="H7107", device_id="BB:BB:BB:BB:BB:BB:BB:BB", device_name="Tower Fan Ranged",
        device_type="devices.types.fan",
        capabilities=[
                    Capability("devices.capabilities.work_mode", "workMode", {
                                                                            "dataType": "STRUCT",
                                                                            "fields": [
                                                                                        {"fieldName": "workMode", "dataType": "ENUM", "options": [
                                                                            {"name": "gearMode", "value": 1},
                                                                                ]},
                    {"fieldName": "modeValue", "dataType": "INTEGER", "range": {"min": 1, "max": 8, "precision": 1}},
                                                    ]})])

    return [bedroom_light_2, bedroom_light, tower_fan, tower_fan_numbered, tower_fan_ranged, thermometer]

def main():
    fake_client = FakeGoveeClient(build_devices())
    tools = GoveeTools(fake_client)

    print("== list_devices ==")
    print(json.dumps(tools.list_devices(), indent=2))

    print("\n== fuzzy name match: 'bedroom light 2' ==")
    print(tools.set_power("bedroom light 2", True))
    assert fake_client.calls[-1] == ("H6006", "56:E2:D0:C9:07:14:6C:40", "devices.capabilities.on_off", "powerSwitch", 1)

    print("\n== ambiguous name: 'light' should raise ==")
    try:
        tools.set_power("light", True)
        raise SystemExit("FAIL: expected DeviceNotFoundError")
    except DeviceNotFoundError as e:
        print(f"OK: {e}")

    print("\n== brightness clamping: 150 -> 100 ==")
    result = tools.set_brightness("Bedroom Light 2", 150)
    print(result)
    assert fake_client.calls[-1][-1] == 100
    assert result["brightness"] == 100, "tool must report the clamped value, not the raw input"

    print("\n== unsupported capability: brightness on thermometer ==")
    result = tools.set_brightness("Wifi Thermometer", 50)
    assert "error" in result
    print(f"OK: {result}")

    print("\n== scene lookup, case-insensitive ==")
    print(tools.set_scene("Bedroom Light 2", "christmas"))
    assert fake_client.calls[-1][-1] == 3065

    print("\n== unknown scene ==")
    result = tools.set_scene("Bedroom Light 2", "Nonexistent Scene")
    assert "error" in result
    print(f"OK: {result}")

    print("\n== fan speed 'medium', named Low/Medium/High options ==")
    print(tools.set_fan_speed("Tower Fan", "medium"))
    assert fake_client.calls[-1][-1] == {"workMode": 1, "modeValue": 2}

    print("\n== fan speed 'high' on UNNAMED numbered gears (1-8, no 'name' key) ==")
    print(tools.set_fan_speed("Tower Fan Numbered", "high"))
    assert fake_client.calls[-1][-1] == {"workMode": 1, "modeValue": 8}

    print("\n== fan speed 'low' on flat INTEGER range work mode ==")
    print(tools.set_fan_speed("Tower Fan Ranged", "low"))
    assert fake_client.calls[-1][-1] == {"workMode": 1, "modeValue": 1}

    print("\n== toggle: oscillation off ==")
    print(tools.set_toggle("Tower Fan", "oscillationToggle", False))
    assert fake_client.calls[-1][-1] == 0

    print("\n== <tool_call> regex parsing ==")
    sample_reply = ("Sure, turning that on.\n<tool_call>\n"'{"name": "set_power", "arguments": {"device_name": "Bedroom Light 2", "on": true}}\n'"</tool_call>")
    matches = TOOL_CALL_RE.findall(sample_reply)
    assert len(matches) == 1
    parsed = json.loads(matches[0])
    assert parsed["name"] == "set_power"
    print(f"OK: parsed {parsed}")

    print("\n== semantic device matching (fake embedding model, no network) ==")
    semantic_match._model = FakeEmbeddingModel()
    try:
        result = tools.get_device_state("the wifi sensor")
        print(f"semantic match result: {result}")
        assert result["device"] == "Wifi Thermometer", "expected semantic fallback to resolve the thermometer"
        try:
            tools.set_power("something with a light in the bedroom", True)
            print("(no exception - acceptable if it happened to resolve to one device)")
        except DeviceNotFoundError as e:
            print(f"OK (ambiguous, correctly declined to guess): {e}")
    finally:
        semantic_match._model = None

    print("\nAll offline checks passed.")

if __name__ == "__main__":
    main()
