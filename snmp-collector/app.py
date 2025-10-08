# app.py
from __future__ import annotations

# --- .env načítanie z rovnakého priečinka ako app.py ---
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    # ak python-dotenv nie je nainštalovaný, skript pobeží ďalej bez neho
    def load_dotenv(*args, **kwargs):
        return False

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

# --- stdlib / third-party ---
import asyncio
import datetime
import signal
from typing import Any, Dict, List, Optional

import pandas as pd

from pysnmp import __version__ as pysnmp_ver
from pyasn1 import __version__ as pyasn1_ver
from pysnmp.entity import config
from pysnmp.hlapi.v3arch.asyncio import (
    SnmpEngine,
    UdpTransportTarget,
    ContextData,
    ObjectType,
    ObjectIdentity,
    UsmUserData,
    get_cmd,
)

# ========================= KONFIGURÁCIA (.env) =========================

def _get_env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if not v:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _parse_hosts(raw: Optional[str]) -> List[str]:
    """
    Akceptuje: '192.168.0.1,192.168.0.2'  alebo  '192.168.0.1 192.168.0.2'
    Ignoruje prázdne položky a úvodzovky.
    """
    if not raw:
        return []
    seps = [",", ";", " ", "\n", "\t", "|"]
    s = str(raw)
    for sp in seps:
        s = s.replace(sp, ",")
    items = []
    for item in s.split(","):
        t = item.strip().strip('"').strip("'")
        if t:
            items.append(t)
    return items

HOSTS: List[str] = _parse_hosts(os.getenv("HOSTS"))

PORT: int = int(os.getenv("PORT", "161"))
TIMEOUT: int = int(os.getenv("TIMEOUT", "5"))
RETRIES: int = int(os.getenv("RETRIES", "2"))
INTERVAL_SECONDS: float = float(os.getenv("INTERVAL_SECONDS", "10"))
FLUSH_EVERY_SECONDS: float = float(os.getenv("FLUSH_EVERY_SECONDS", "30"))
PRINT_EVERY_CYCLES: int = int(os.getenv("PRINT_EVERY_CYCLES", "3"))

SNMP_USER: str = os.getenv("SNMP_USER", "")
SNMP_AUTH_KEY: str = os.getenv("SNMP_AUTH_KEY", "")
SNMP_PRIV_KEY: str = os.getenv("SNMP_PRIV_KEY", "")
AUTH_PROTO: str = os.getenv("AUTH_PROTO", "USM_AUTH_HMAC96_SHA")
PRIV_PROTO: str = os.getenv("PRIV_PROTO", "USM_PRIV_CFB128_AES")

USE_SQL_FLUSH: bool = _get_env_bool("USE_SQL_FLUSH", False)
SQL_SERVER: str = os.getenv("SQL_SERVER", "")
SQL_DATABASE: str = os.getenv("SQL_DATABASE", "")
SQL_USERNAME: str = os.getenv("SQL_USERNAME", "")
SQL_PASSWORD: str = os.getenv("SQL_PASSWORD", "")
SQL_DRIVER: str = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")
SQL_TABLE: str = os.getenv("SQL_TABLE", "SNMP_Telemetry")
SQL_SCHEMA: str = os.getenv("SQL_SCHEMA", "dbo")
SQL_ENCRYPT: str = os.getenv("SQL_ENCRYPT", "no")  # "yes"|"no"
SQL_TRUSTCERT: str = os.getenv("SQL_TRUSTCERT", "yes")  # "yes"|"no"

# ------------------------- sanity printy -------------------------
try:
    import Cryptodome  # pycryptodomex
    _crypto_ok = True
except Exception:
    _crypto_ok = False

print(f"[SANITY] pycryptodomex {'OK' if _crypto_ok else 'MISSING'} | "
        f"pysnmp={pysnmp_ver} | pyasn1={pyasn1_ver}")

print(f"[SANITY] pysnmp host list -> {HOSTS if HOSTS else []}")

# ------------------------- SNMP mapy -------------------------
AUTH_MAP: Dict[str, Any] = {
    "USM_AUTH_HMAC96_MD5":     config.USM_AUTH_HMAC96_MD5,
    "USM_AUTH_HMAC96_SHA":     config.USM_AUTH_HMAC96_SHA,
    "USM_AUTH_HMAC128_SHA224": config.USM_AUTH_HMAC128_SHA224,
    "USM_AUTH_HMAC192_SHA256": config.USM_AUTH_HMAC192_SHA256,
    "USM_AUTH_HMAC256_SHA384": config.USM_AUTH_HMAC256_SHA384,
    "USM_AUTH_HMAC384_SHA512": config.USM_AUTH_HMAC384_SHA512,
}
PRIV_MAP: Dict[str, Any] = {
    "USM_PRIV_CFB128_AES": config.USM_PRIV_CFB128_AES,
    "USM_PRIV_CFB192_AES": config.USM_PRIV_CFB192_AES,
    "USM_PRIV_CFB256_AES": config.USM_PRIV_CFB256_AES,
    "USM_PRIV_CBC56_DES":  config.USM_PRIV_CBC56_DES,
}

METRICS: Dict[str, str] = {
    "System Name":               "1.3.6.1.2.1.1.5.0",
    "System Description":        "1.3.6.1.2.1.1.1.0",
    "System Uptime (TimeTicks)": "1.3.6.1.2.1.1.3.0",
    "System Contact":            "1.3.6.1.2.1.1.4.0",
    "System Location":           "1.3.6.1.2.1.1.6.0",
    "System Services":           "1.3.6.1.2.1.1.7.0",
    "CPU Load 1min":             "1.3.6.1.4.1.2021.10.1.3.1",
    "CPU Load 5min":             "1.3.6.1.4.1.2021.10.1.3.2",
    "CPU Load 15min":            "1.3.6.1.4.1.2021.10.1.3.3",
    "Total RAM (kB)":            "1.3.6.1.4.1.2021.4.5.0",
    "Available RAM (kB)":        "1.3.6.1.4.1.2021.4.6.0",
    "Used RAM (kB)":             "1.3.6.1.4.1.2021.4.11.0",
    "Buffer RAM (kB)":           "1.3.6.1.4.1.2021.4.14.0",
}

def _proto(name: str, mapping: Dict[str, Any]):
    key = (name or "").strip().upper()
    if key not in mapping:
        raise ValueError(f"Unknown protocol '{name}'.")
    return mapping[key]

def _user() -> UsmUserData:
    return UsmUserData(
        SNMP_USER,
        SNMP_AUTH_KEY,
        SNMP_PRIV_KEY,
        authProtocol=_proto(AUTH_PROTO, AUTH_MAP),
        privProtocol=_proto(PRIV_PROTO, PRIV_MAP),
    )

# ------------------------- helpery -------------------------
def _safe_float(x: Optional[str]) -> float:
    try:
        return float(str(x))
    except Exception:
        return 0.0

async def _get_scalar(engine: SnmpEngine, target: UdpTransportTarget, oid: str) -> Optional[str]:
    err_ind, err_stat, _, vbs = await get_cmd(
        engine, _user(), target, ContextData(), ObjectType(ObjectIdentity(oid))
    )
    if err_ind or err_stat or not vbs:
        return None
    return str(vbs[0][1])

def _fail_row(host: str, err: str) -> Dict[str, Any]:
    row = {
        "run_ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": host,
        "error": err,
        "RAM Free (%)": None,
    }
    for k in METRICS:
        row[k] = "NO_DATA"
    return row

# ------------------------- SQL flush -------------------------
def flush_to_sql(df: pd.DataFrame):
    if not USE_SQL_FLUSH or df.empty:
        return
    try:
        from sqlalchemy import create_engine
        # mssql+pyodbc URL – driver musí byť URL-enkódovaný
        from urllib.parse import quote_plus

        params = {
            "driver": SQL_DRIVER,
            "Encrypt": SQL_ENCRYPT,
            "TrustServerCertificate": SQL_TRUSTCERT,
        }
        query = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())

        conn_str = (
            f"mssql+pyodbc://{quote_plus(SQL_USERNAME)}:{quote_plus(SQL_PASSWORD)}"
            f"@{SQL_SERVER}/{quote_plus(SQL_DATABASE)}?{query}"
        )
        engine = create_engine(conn_str, fast_executemany=True)
        df.to_sql(SQL_TABLE, con=engine, schema=SQL_SCHEMA, index=False, if_exists="append")
        print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] Flushed {len(df)} rows to SQL.")
    except Exception as e:
        print(f"[SQL] Flush failed: {e}")

# ------------------------- SNMP poll funkcie -------------------------
async def poll_host(engine: SnmpEngine, target: UdpTransportTarget, host: str) -> Dict[str, Any]:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    row: Dict[str, Any] = {"run_ts": ts, "host": host, "error": None}
    try:
        # rýchly sanity check – sysName
        sysname = await _get_scalar(engine, target, METRICS["System Name"])
        if sysname is None:
            return _fail_row(host, "No SNMP response (sysName)")

        # všetky metriky
        for label, oid in METRICS.items():
            v = await _get_scalar(engine, target, oid)
            row[label] = v if v is not None else "NO_DATA"

        # RAM Free (%)
        total_kb = _safe_float(row.get("Total RAM (kB)"))
        avail_kb = _safe_float(row.get("Available RAM (kB)"))
        row["RAM Free (%)"] = round((avail_kb / total_kb) * 100, 2) if total_kb > 0 else None

        return row
    except Exception as e:
        return _fail_row(host, f"{type(e).__name__}: {e}")

async def poll_once(engine: SnmpEngine, targets: Dict[str, UdpTransportTarget]) -> pd.DataFrame:
    tasks = [poll_host(engine, targets[h], h) for h in targets]
    rows = await asyncio.gather(*tasks)
    df = pd.DataFrame(rows)
    preferred = ["run_ts", "host"] + list(METRICS.keys()) + ["RAM Free (%)", "error"]
    df = df[[c for c in preferred if c in df.columns]]
    return df

# =============================== MAIN ===============================
async def main():
    if not HOSTS:
        print("No valid HOSTS parsed from .env. Nastav HOSTS bez úvodzoviek a bez koncovej čiarky.")
        raise SystemExit("No valid HOSTS parsed from .env. Nastav HOSTS bez úvodzoviek a bez koncovej čiarky.")

    print("Starting SNMP poller … Ctrl+C to stop.")

    engine = SnmpEngine()

    # jeden transport target na host
    targets: Dict[str, UdpTransportTarget] = {}
    for h in HOSTS:
        targets[h] = await UdpTransportTarget.create((h, PORT), timeout=TIMEOUT, retries=RETRIES)

    # graceful shutdown
    stop_event = asyncio.Event()

    def _stop():
        stop_event.set()

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig:
            try:
                asyncio.get_running_loop().add_signal_handler(sig, _stop)
            except NotImplementedError:
                # Windows bez podpory signal handlerov
                pass

    buffer_frames: List[pd.DataFrame] = []
    last_flush = asyncio.get_event_loop().time()
    cycle = 0

    try:
        while not stop_event.is_set():
            cycle_start = asyncio.get_event_loop().time()

            df_cycle = await poll_once(engine, targets)
            buffer_frames.append(df_cycle)

            cycle += 1
            if PRINT_EVERY_CYCLES > 0 and (cycle % PRINT_EVERY_CYCLES == 0):
                try:
                    print(df_cycle.to_string(index=False))
                except Exception:
                    print(df_cycle.head())

            now = asyncio.get_event_loop().time()
            if (now - last_flush) >= FLUSH_EVERY_SECONDS:
                df_to_flush = pd.concat(buffer_frames, ignore_index=True) if buffer_frames else pd.DataFrame()
                if not df_to_flush.empty:
                    flush_to_sql(df_to_flush)
                buffer_frames.clear()
                last_flush = now

            elapsed = asyncio.get_event_loop().time() - cycle_start
            await asyncio.sleep(max(0, INTERVAL_SECONDS - elapsed))

    finally:
        # korektné zatvorenie transportov a dispatcheru
        for t in targets.values():
            try:
                await t.close()
            except Exception:
                pass
        try:
            engine.transportDispatcher.closeDispatcher()
        except Exception:
            pass
        # posledný flush bufferu
        if buffer_frames:
            try:
                df_to_flush = pd.concat(buffer_frames, ignore_index=True)
                flush_to_sql(df_to_flush)
            except Exception:
                pass
        print("SNMP poller stopped cleanly.")

if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except Exception:
        pass
    asyncio.run(main())