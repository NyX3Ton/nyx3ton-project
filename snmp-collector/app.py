from __future__ import annotations
import os, asyncio, datetime, signal, urllib.parse
from typing import Any, Dict, List, Optional
import pandas as pd

from pysnmp.entity import config
from pysnmp.hlapi.v3arch.asyncio import (
    SnmpEngine, UdpTransportTarget, ContextData,
    ObjectType, ObjectIdentity, UsmUserData, get_cmd
)

def must_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        sys.exit(f"Missing required environment variable: {name}")
    return v

HOSTS = must_env("HOSTS").split(",")
SNMP_USER     = must_env("SNMP_USER")
SNMP_AUTH_KEY = must_env("SNMP_AUTH_KEY")
SNMP_PRIV_KEY = must_env("SNMP_PRIV_KEY")
AUTH_PROTO    = os.getenv("AUTH_PROTO", "USM_AUTH_HMAC96_SHA")
PRIV_PROTO    = os.getenv("PRIV_PROTO", "USM_PRIV_CFB128_AES")

PORT                = int(os.getenv("PORT", "161"))
TIMEOUT             = int(os.getenv("TIMEOUT", "10"))
RETRIES             = int(os.getenv("RETRIES", "2"))
INTERVAL_SECONDS    = int(os.getenv("INTERVAL_SECONDS", "10"))
PRINT_EVERY_CYCLES  = int(os.getenv("PRINT_EVERY_CYCLES", "3"))
FLUSH_EVERY_SECONDS = int(os.getenv("FLUSH_EVERY_SECONDS", "60"))

USE_SQL_FLUSH = os.getenv("USE_SQL_FLUSH", "false").lower() == "true"
SQL_SERVER    = must_env("SQL_SERVER") if USE_SQL_FLUSH else ""
SQL_DATABASE  = os.getenv("SQL_DATABASE", "")
SQL_USERNAME  = os.getenv("SQL_USERNAME", "")
SQL_PASSWORD  = os.getenv("SQL_PASSWORD", "")
SQL_DRIVER    = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")
SQL_TABLE     = os.getenv("SQL_TABLE", "SNMP_Telemetry")
SQL_SCHEMA    = os.getenv("SQL_SCHEMA", "dbo")

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

def _proto(name: str, mapping: Dict[str, Any]):
    return mapping[name.upper()]

def _user() -> UsmUserData:
    return UsmUserData(
        SNMP_USER, SNMP_AUTH_KEY, SNMP_PRIV_KEY,
        authProtocol=_proto(AUTH_PROTO, AUTH_MAP),
        privProtocol=_proto(PRIV_PROTO, PRIV_MAP),
    )

def _safe_float(x: Optional[str]) -> float:
    try: return float(str(x))
    except Exception: return 0.0

async def _get_scalar(engine: SnmpEngine, target: UdpTransportTarget, oid: str) -> Optional[str]:
    err_ind, err_stat, _, vbs = await get_cmd(engine, _user(), target, ContextData(),
                                                ObjectType(ObjectIdentity(oid)))
    if err_ind or err_stat or not vbs: return None
    return str(vbs[0][1])

def _fail_row(host: str, err: str) -> Dict[str, Any]:
    row = {"run_ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": host, "error": err, "RAM Free (%)": None}
    for k in METRICS: row[k] = "NO_DATA"
    return row

async def poll_host(engine: SnmpEngine, target: UdpTransportTarget, host: str) -> Dict[str, Any]:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    row: Dict[str, Any] = {"run_ts": ts, "host": host, "error": None}
    try:
        if await _get_scalar(engine, target, METRICS["System Name"]) is None:
            return _fail_row(host, "No SNMP response (sysName)")
        for label, oid in METRICS.items():
            v = await _get_scalar(engine, target, oid)
            row[label] = v if v is not None else "NO_DATA"
        total_kb  = _safe_float(row.get("Total RAM (kB)"))
        avail_kb  = _safe_float(row.get("Available RAM (kB)"))
        row["RAM Free (%)"] = round((avail_kb / total_kb) * 100, 2) if total_kb > 0 else None
        return row
    except Exception as e:
        return _fail_row(host, f"{type(e).__name__}: {e}")

async def poll_once(engine: SnmpEngine, targets: Dict[str, UdpTransportTarget]) -> pd.DataFrame:
    rows = await asyncio.gather(*[poll_host(engine, targets[h], h) for h in targets])
    df = pd.DataFrame(rows)
    preferred = ["run_ts", "host"] + list(METRICS.keys()) + ["RAM Free (%)", "error"]
    return df[[c for c in preferred if c in df.columns]]

def flush_to_sql(df: pd.DataFrame):
    if not USE_SQL_FLUSH or df.empty: return
    try:
        from sqlalchemy import create_engine
        dsn = (
            f"DRIVER={{{SQL_DRIVER}}};"
            f"SERVER={SQL_SERVER};"
            f"DATABASE={SQL_DATABASE};"
            f"UID={SQL_USERNAME};PWD={SQL_PASSWORD};"
            "Encrypt=no;TrustServerCertificate=yes"
        )
        params = urllib.parse.quote_plus(dsn)
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)
        df.to_sql(SQL_TABLE, con=engine, schema=SQL_SCHEMA, index=False, if_exists="append")
        print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] Flushed {len(df)} rows to SQL.")
    except Exception as e:
        print(f"[SQL] Flush failed: {e}")

async def main():
    print("Starting SNMP poller … Ctrl+C to stop.")
    engine = SnmpEngine()
    targets: Dict[str, UdpTransportTarget] = {
        h: await UdpTransportTarget.create((h, PORT), timeout=TIMEOUT, retries=RETRIES)
        for h in HOSTS
    }

    stop_event = asyncio.Event()
    def _stop(): stop_event.set()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig:
            try: asyncio.get_running_loop().add_signal_handler(sig, _stop)
            except NotImplementedError: pass

    buffer_frames: List[pd.DataFrame] = []
    last_flush = asyncio.get_event_loop().time()
    cycle = 0

    try:
        while not stop_event.is_set():
            cycle_start = asyncio.get_event_loop().time()
            df_cycle = await poll_once(engine, targets)
            buffer_frames.append(df_cycle)
            cycle += 1
            if cycle % PRINT_EVERY_CYCLES == 0:
                try: print(df_cycle.to_string(index=False))
                except Exception: print(df_cycle.head())
            now = asyncio.get_event_loop().time()
            if (now - last_flush) >= FLUSH_EVERY_SECONDS:
                df_to_flush = pd.concat(buffer_frames, ignore_index=True) if buffer_frames else pd.DataFrame()
                if not df_to_flush.empty: flush_to_sql(df_to_flush)
                buffer_frames.clear(); last_flush = now
            elapsed = asyncio.get_event_loop().time() - cycle_start
            await asyncio.sleep(max(0, INTERVAL_SECONDS - elapsed))
    finally:
        for t in targets.values():
            try: await t.close()
            except Exception: pass
        try: engine.transportDispatcher.closeDispatcher()
        except Exception: pass
        if buffer_frames:
            try: flush_to_sql(pd.concat(buffer_frames, ignore_index=True))
            except Exception: pass
        print("SNMP poller stopped cleanly.")

if __name__ == "__main__":
    try:
        import nest_asyncio; nest_asyncio.apply()
    except Exception: pass
    asyncio.run(main())
