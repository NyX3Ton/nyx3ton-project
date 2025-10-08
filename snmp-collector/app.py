from __future__ import annotations
import asyncio, datetime, signal
from typing import Any, Dict, List, Optional
import pandas as pd

from pysnmp.entity import config
from pysnmp.hlapi.v3arch.asyncio import (
    SnmpEngine, UdpTransportTarget, ContextData,
    ObjectType, ObjectIdentity, UsmUserData, get_cmd
)

# ================== KONFIGURÁCIA ==================
HOSTS = ["192.168.50.159"]

SNMP_USER     = "gpolak"
SNMP_AUTH_KEY = "W4d3r100@"
SNMP_PRIV_KEY = "W4d3r100@"

AUTH_PROTO = "USM_AUTH_HMAC96_SHA"
PRIV_PROTO = "USM_PRIV_CFB128_AES"

PORT, TIMEOUT, RETRIES = 161, 10, 2
INTERVAL_SECONDS       = 5
FLUSH_EVERY_SECONDS    = 600
PRINT_EVERY_CYCLES     = 3   # vypíš každú 6-tu iteráciu (~30 s pri 5 s intervale)

USE_SQL_FLUSH = False
SQL_SERVER   = "192.168.50.88"
SQL_DATABASE = "data_science"
SQL_USERNAME = "sa"
SQL_PASSWORD = "W4rpDr1v3@"
SQL_DRIVER   = "ODBC Driver 17 for SQL Server"
SQL_TABLE    = "SNMP_Telemetry"
SQL_SCHEMA   = "dbo"

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

async def poll_host(engine: SnmpEngine, target: UdpTransportTarget, host: str) -> Dict[str, Any]:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    row: Dict[str, Any] = {"run_ts": ts, "host": host, "error": None}
    try:
        # rýchly test – sysName
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

def flush_to_sql(df: pd.DataFrame):
    if not USE_SQL_FLUSH or df.empty:
        return
    try:
        from sqlalchemy import create_engine
        conn_str = (
            f"mssql+pyodbc://{SQL_USERNAME}:{SQL_PASSWORD}"
            f"@{SQL_SERVER}/{SQL_DATABASE}"
            f"?driver={SQL_DRIVER}&Encrypt=no&TrustServerCertificate=yes"
        )
        engine = create_engine(conn_str, fast_executemany=True)
        df.to_sql(SQL_TABLE, con=engine, schema=SQL_SCHEMA, index=False, if_exists="append")
        print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] Flushed {len(df)} rows to SQL.")
    except Exception as e:
        print(f"[SQL] Flush failed: {e}")

async def main():
    print("Starting SNMP poller … Ctrl+C to stop.")

    # 1) Jeden engine na celý beh
    engine = SnmpEngine()

    # 2) Jeden transport target na host – recyklovať!
    targets: Dict[str, UdpTransportTarget] = {}
    for h in HOSTS:
        targets[h] = await UdpTransportTarget.create((h, PORT), timeout=TIMEOUT, retries=RETRIES)

    # Graceful shutdown
    stop_event = asyncio.Event()
    def _handle_sig():
        stop_event.set()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig:
            try:
                asyncio.get_running_loop().add_signal_handler(sig, _handle_sig)
            except NotImplementedError:
                # Windows: ignore
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
            if cycle % PRINT_EVERY_CYCLES == 0:
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
        # korektné zatvorenie transportov a dispatcheru – zabráni únikom FD
        for t in targets.values():
            try:
                await t.close()
            except Exception:
                pass
        try:
            # zavrie interný asyncio dispatcher
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
        import nest_asyncio; nest_asyncio.apply()
    except Exception:
        pass
    asyncio.run(main())