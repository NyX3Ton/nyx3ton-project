import os, io, json, base64, datetime as dt
import requests
import pandas as pd
import matplotlib.pyplot as plt
import asyncio, websockets

# ---------- Config (override via env if needed) ----------
NIM_BASE = os.getenv("NIM_BASE", "http://localhost:8000/v1")
NIM_KEY  = os.getenv("NIM_KEY",  "nim-local-key")         # vLLM usually ignores this
MCP_URL  = os.getenv("MCP_URL",  "ws://localhost:5173")

# A tiny task + schema hint so the model writes decent SQL
TASK = os.getenv("TASK", "Show weekly active users over the last 12 weeks.")
SCHEMA_HINT = os.getenv("SCHEMA_HINT",
    "dbo.Users(id INT, created_at DATETIME2, status NVARCHAR(20)); "
    "active users have status='active'.")

# Tool name in your MCP server. Common values:
#   "mssql.run_sql", "sql.query", or "db.query".
MCP_SQL_TOOL = os.getenv("MCP_SQL_TOOL", "mssql.run_sql")

# ---------- LLM helper (OpenAI-compatible /chat/completions) ----------
def llm_chat(messages, model="nemotron", temperature=0.2):
    r = requests.post(
        f"{NIM_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {NIM_KEY}"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=120
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ---------- MCP helpers ----------
async def mcp_call(tool_name: str, args: dict):
    async with websockets.connect(MCP_URL, ping_interval=20) as ws:
        # (Optional) list tools once; not strictly required
        await ws.send(json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/list"}))
        _ = await ws.recv()

        await ws.send(json.dumps({
            "jsonrpc":"2.0",
            "id":2,
            "method":"tools/call",
            "params":{"name": tool_name, "arguments": args}
        }))
        raw = await ws.recv()
        msg = json.loads(raw)
        if "error" in msg:
            raise RuntimeError(msg["error"])
        return msg["result"]

def run_sql(sql: str) -> pd.DataFrame:
    result = asyncio.get_event_loop().run_until_complete(
        mcp_call(MCP_SQL_TOOL, {"sql": sql})
    )
    # Expect either {columns: [...], rows: [...]} or {schema: [{name:..}], rows: [...]}
    if "columns" in result:
        cols = result["columns"]
    elif "schema" in result:
        cols = [c["name"] for c in result["schema"]]
    else:
        raise ValueError(f"Unexpected MCP result shape: {result.keys()}")
    return pd.DataFrame(result.get("rows", []), columns=cols)

# ---------- plotting ----------
def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def simple_chart(df: pd.DataFrame) -> tuple[str, str]:
    # Heuristic: if there's a date-like column, do time series; else bar of first two cols.
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ("date","time","created_at","timestamp"))]
    if date_cols:
        x = date_cols[0]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        y = num_cols[0] if num_cols else df.columns[-1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pd.to_datetime(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce"))
        ax.set_title(f"{y} over {x}")
        ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(True, alpha=0.3)
        return fig_to_base64(fig), f"Time series of {y} over {x}."
    else:
        x = df.columns[0]
        y = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(df[x].astype(str).head(20), pd.to_numeric(df[y], errors="coerce").head(20))
        ax.set_title(f"{y} by {x} (top 20)")
        ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        return fig_to_base64(fig), f"Bar chart of {y} by {x} (top 20)."

# ---------- main ----------
def main():
    # 1) Get SQL from Nemotron
    system = {"role":"system","content":(
        "You write a SINGLE Microsoft SQL Server SELECT query for the given task and schema. "
        "Use ISO dates; prefer weekly aggregation; no DDL/DML; keep it efficient. "
        "Return ONLY the SQL on one line."
    )}
    user = {"role":"user","content":f"Task: {TASK}\nSchema: {SCHEMA_HINT}"}
    sql = llm_chat([system, user]).strip().strip("`").strip()
    if not sql.lower().startswith("select"):
        raise ValueError(f"Refusing non-SELECT SQL: {sql}")

    # 2) Execute via MCP
    df = run_sql(sql)
    if df.empty:
        raise RuntimeError("Query returned no rows. Try adjusting TASK/SCHEMA_HINT.")

    # 3) Visualize
    img_b64, caption = simple_chart(df)

    # 4) Write HTML
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Auto Report</title>
<style>body{{font-family:system-ui,Segoe UI,Roboto,Arial;margin:24px}}h1{{margin:4px 0}}.muted{{color:#666}}</style>
</head><body>
<h1>Automated Report</h1>
<p class="muted">{now}</p>
<h2>Task</h2><pre>{TASK}</pre>
<h2>SQL</h2><pre>{sql}</pre>
<h2>Visualization</h2>
<p><img alt="chart" src="data:image/png;base64,{img_b64}" /></p>
<p class="muted">{caption}</p>
<h2>Data preview</h2>
<pre>{df.head(20).to_markdown(index=False)}</pre>
</body></html>"""
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Wrote report.html")

if __name__ == "__main__":
    main()
