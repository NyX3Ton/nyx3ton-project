"""Container entrypoint.

`app.py` is imported UNMODIFIED. It hardcodes `demo.launch(server_name="127.0.0.1")`,
which is fine on a desktop but unreachable from outside a container. So instead of
running `python app.py`, we import it (which loads the datasets and trains the models
at module level, exactly as normal) and then launch the Gradio app bound to 0.0.0.0
so Docker's published ports work.

app.py is never edited — only imported.
"""
from __future__ import annotations

import os
import threading

import app  # importing runs dataset load + model training at module level


def _start_optuna_dashboard() -> None:
    """Start the Optuna dashboard bound to 0.0.0.0 (app.launch_optuna_dashboard binds 127.0.0.1)."""
    if os.getenv("OPTUNA_DASHBOARD", "1") != "1":
        return
    try:
        from optuna_dashboard import run_server
    except ImportError:
        print("optuna-dashboard not installed; dashboard disabled.")
        return
    port = int(os.getenv("OPTUNA_DASHBOARD_PORT", "8080"))

    def _run() -> None:
        try:
            run_server(app.OPTUNA_STORAGE, host="0.0.0.0", port=port)
        except Exception as exc:  # noqa: BLE001
            print(f"Optuna dashboard failed to start: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    print(f"Optuna dashboard listening on 0.0.0.0:{port}")


if __name__ == "__main__":
    _start_optuna_dashboard()
    app.demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        inbrowser=False,
        share=False,
        css=app.CUSTOM_CSS,
    )
