#weather_client.py

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from . import config

logger = logging.getLogger("weather_client")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo reports weather as a numeric WMO code with no label attached -
# this maps the common codes to a short human-readable condition string.
# https://open-meteo.com/en/docs#weathervariables
_WMO_CONDITIONS = {
                    0: "Clear sky", 
                    1: "Mainly clear", 
                    2: "Partly cloudy", 
                    3: "Overcast",
                    45: "Fog", 
                    48: "Depositing rime fog",
                    51: "Light drizzle", 
                    53: "Moderate drizzle", 
                    55: "Dense drizzle",
                    56: "Light freezing drizzle", 
                    57: "Dense freezing drizzle",
                    61: "Slight rain", 
                    63: "Moderate rain", 
                    65: "Heavy rain",
                    66: "Light freezing rain", 
                    67: "Heavy freezing rain",
                    71: "Slight snow fall", 
                    73: "Moderate snow fall", 
                    75: "Heavy snow fall",
                    77: "Snow grains",
                    80: "Slight rain showers", 
                    81: "Moderate rain showers", 
                    82: "Violent rain showers",
                    85: "Slight snow showers", 
                    86: "Heavy snow showers",
                    95: "Thunderstorm", 
                    96: "Thunderstorm with slight hail", 
                    99: "Thunderstorm with heavy hail",
                    }


def _condition_for(code: Optional[int]) -> str:
    if code is None:
        return "Unknown"
    return _WMO_CONDITIONS.get(int(code), f"Unknown (code {code})")

class WeatherError(RuntimeError):
    pass

def _default_get_json(url: str, params: dict) -> dict:
    import requests

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

class WeatherClient:
    def __init__(self, get_json_fn: Optional[Callable[[str, dict], dict]] = None):
        self._get_json = get_json_fn or _default_get_json
        self._geocode_cache: dict[str, tuple[float, float, str]] = {}

    def _geocode(self, location: str) -> tuple[float, float, str]:
        key = location.strip().lower()
        if key in self._geocode_cache:
            return self._geocode_cache[key]

        try:
            data = self._get_json(GEOCODE_URL, {"name": location, "count": 1})
        except Exception as e:  # noqa: BLE001
            raise WeatherError(f"Couldn't look up location '{location}': {e}") from e

        results = data.get("results") or []
        if not results:
            raise WeatherError(f"No location found matching '{location}'")

        top = results[0]
        lat, lon = top["latitude"], top["longitude"]
        name_parts = [top.get("name")]
        if top.get("admin1"):
            name_parts.append(top["admin1"])
        if top.get("country"):
            name_parts.append(top["country"])
        resolved_name = ", ".join(p for p in name_parts if p)

        self._geocode_cache[key] = (lat, lon, resolved_name)
        return lat, lon, resolved_name

    def get_forecast(self, location: Optional[str] = None) -> dict:
        location = (location or config.GOVEE_DEFAULT_LOCATION).strip()
        if not location:
            raise WeatherError(
                                "No location given and GOVEE_DEFAULT_LOCATION isn't set - "
                                "specify a city or set a default location."
                            )

        lat, lon, resolved_name = self._geocode(location)

        try:
            data = self._get_json(FORECAST_URL, {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                "forecast_days": 3,
                "timezone": "auto",
            })
        except Exception as e:  # noqa: BLE001
            raise WeatherError(f"Couldn't fetch forecast for '{resolved_name}': {e}") from e

        current = data.get("current", {})
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        codes = daily.get("weather_code", [])

        forecast = [
                    {
                    "date": dates[i],
                    "high_c": highs[i] if i < len(highs) else None,
                    "low_c": lows[i] if i < len(lows) else None,
                    "condition": _condition_for(codes[i] if i < len(codes) else None),
                    }
            for i in range(len(dates))
                    ]

        return {
                "location": resolved_name,
                "temperature_c": current.get("temperature_2m"),
                "condition": _condition_for(current.get("weather_code")),
                "humidity_pct": current.get("relative_humidity_2m"),
                "wind_kph": current.get("wind_speed_10m"),
                "forecast": forecast,
                }
