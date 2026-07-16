#govee_client.py

from __future__ import annotations

import os
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import requests

logger = logging.getLogger("govee_client")

BASE_URL = "https://openapi.api.govee.com"
DEVICES_PATH = "/router/api/v1/user/devices"
CONTROL_PATH = "/router/api/v1/device/control"
STATE_PATH = "/router/api/v1/device/state"
class GoveeAPIError(RuntimeError):

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Govee API error {status_code}: {message}")

class GoveeRateLimitError(GoveeAPIError):
    pass
@dataclass
class Capability:
    type: str
    instance: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def data_type(self) -> str:
        return self.parameters.get("dataType", "")

    @property
    def options(self) -> list[dict[str, Any]]:
        """For ENUM capabilities: [{'name': 'on', 'value': 1}, ...]"""
        return self.parameters.get("options", [])

    @property
    def value_range(self) -> Optional[dict[str, Any]]:
        """For INTEGER capabilities: {'min': .., 'max': .., 'precision': ..}"""
        return self.parameters.get("range")
@dataclass
class Device:
    sku: str
    device_id: str
    device_name: str
    device_type: str
    capabilities: list[Capability]

    def capability(self, cap_type: str, instance: str) -> Optional[Capability]:
        for c in self.capabilities:
            if c.type == cap_type and c.instance == instance:
                return c
        return None

    def has(self, cap_type: str, instance: str) -> bool:
        return self.capability(cap_type, instance) is not None
class GoveeClient:
    def __init__(self,api_key: Optional[str] = None,timeout: float = 10.0,verify: Union[bool, str] = True):

        self.api_key = api_key or os.environ.get("GOVEE_API_KEY")
        if not self.api_key:
            raise ValueError("Govee API key not provided (arg or GOVEE_API_KEY env var)")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = verify
        self.session.headers.update({"Govee-API-Key": self.api_key,"Content-Type": "application/json"})
        self._device_cache: Optional[list[Device]] = None

    def _request(self, method: str, path: str, json_body: Optional[dict] = None) -> dict:
        url = f"{BASE_URL}{path}"
        resp = self.session.request(method, url, json=json_body, timeout=self.timeout)

        remaining = resp.headers.get("X-RateLimit-Remaining") or resp.headers.get("API-RateLimit-Remaining")
        if remaining is not None:
            logger.debug("Govee rate limit remaining: %s", remaining)

        if resp.status_code == 429:
            raise GoveeRateLimitError(429, "Rate limit exceeded (10,000 requests/account/day)")
        if resp.status_code == 401:
            raise GoveeAPIError(401, "Unauthorized - check GOVEE_API_KEY")
        if resp.status_code >= 400:
            try:
                body = resp.json()
                msg = body.get("message") or body.get("msg") or resp.text
            except ValueError:
                msg = resp.text
            raise GoveeAPIError(resp.status_code, msg)

        return resp.json()

    def list_devices(self, force_refresh: bool = False) -> list[Device]:
        if self._device_cache is not None and not force_refresh:
            return self._device_cache

        data = self._request("GET", DEVICES_PATH)
        devices: list[Device] = []
        for raw in data.get("data", []):
            caps = [
                    Capability(type=c["type"], instance=c["instance"], parameters=c.get("parameters", {}))
                    for c in raw.get("capabilities", [])
                    ]
            devices.append(Device(
                                    sku=raw["sku"],
                                    device_id=raw["device"],
                                    device_name=raw.get("deviceName", raw["sku"]),
                                    device_type=raw.get("type", "devices.types.light"),
                                    capabilities=caps
                                    ))
        self._device_cache = devices
        return devices

    def get_device(self, device_id: str) -> Device:
        for d in self.list_devices():
            if d.device_id == device_id:
                return d
        raise KeyError(f"No cached device with id {device_id}. Try list_devices(force_refresh=True).")

    def get_state(self, sku: str, device_id: str) -> dict[str, Any]:

        body = {"requestId": str(uuid.uuid4()),"payload": {"sku": sku, "device": device_id}}
        data = self._request("POST", STATE_PATH, body)
        payload = data.get("payload", {})
        state: dict[str, Any] = {}
        for cap in payload.get("capabilities", []):
            instance = cap.get("instance")
            value = cap.get("state", {}).get("value")
            if cap.get("type") == "devices.capabilities.online":
                state["online"] = bool(value)
            elif instance == "sensorTemperature" and isinstance(value, (int, float)):
                state[instance] = round((value - 32) * 5 / 9, 1)
            else:
                state[instance] = value
        return state

    def control(self, sku: str, device_id: str, cap_type: str, instance: str, value: Any) -> dict:
        body = {
                "requestId": str(uuid.uuid4()),
                "payload": {"sku": sku,"device": device_id,"capability": {"type": cap_type, "instance": instance, "value": value}}
                }
        return self._request("POST", CONTROL_PATH, body)

    def set_power(self, device: Device, on: bool) -> dict:
        return self.control(device.sku, device.device_id, "devices.capabilities.on_off", "powerSwitch", 1 if on else 0)

    def set_brightness(self, device: Device, percent: int) -> dict:
        cap = device.capability("devices.capabilities.range", "brightness")
        if cap and cap.value_range:
            lo, hi = cap.value_range["min"], cap.value_range["max"]
            percent = max(lo, min(hi, percent))
        return self.control(device.sku, device.device_id, "devices.capabilities.range", "brightness", percent)

    def set_color_rgb(self, device: Device, r: int, g: int, b: int) -> dict:
        rgb_int = (r << 16) + (g << 8) + b
        return self.control(device.sku, device.device_id, "devices.capabilities.color_setting", "colorRgb", rgb_int)

    def set_color_temp(self, device: Device, kelvin: int) -> dict:
        cap = device.capability("devices.capabilities.color_setting", "colorTemperatureK")
        if cap and cap.value_range:
            lo, hi = cap.value_range["min"], cap.value_range["max"]
            kelvin = max(lo, min(hi, kelvin))
        return self.control(device.sku, device.device_id, "devices.capabilities.color_setting", "colorTemperatureK", kelvin)

    def set_scene(self, device: Device, scene_value: int, instance: str = "lightScene") -> dict:
        return self.control(device.sku, device.device_id, "devices.capabilities.dynamic_scene", instance, scene_value)
