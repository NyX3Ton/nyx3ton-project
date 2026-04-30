# main script: app-few_shot.py of nyx3ton-project\CV Validator

from __future__ import annotations

# -----------------------------------------------------------------------------
# 0. IMPORTS
# 1. GLOBAL MODEL CACHE
# 2. EXTERNAL HELPERS
# 3. DOCUMENT LOADING: PDF/DOCX/RTF/TXT/DOC
# 4. JOB AD SCRAPING
# 5. CHUNKING + RAG
# 6. REPORTING
# 7. MAIN PIPELINE
# 8. GRADIO UI
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 0. IMPORTS
# -----------------------------------------------------------------------------
import gc, json, os, re, tempfile, traceback, torch, requests, faiss
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import numpy as np
import gradio as gr
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from dictionary_fallback import (fallback_extract_requirements_from_text, build_hybrid_requirement_result)

# -----------------------------------------------------------------------------
# 1. ENV + GLOBAL SETTINGS
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

DEFAULT_EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID","sentence-transformers/paraphrase-multilingual-mpnet-base-v2",)
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "220"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "55"))
DEFAULT_TOP_K = max(2, min(20, int(os.getenv("TOP_K", "5"))))
DEFAULT_MAX_REQUIREMENTS = max(5, min(25, int(os.getenv("MAX_REQUIREMENTS", "12"))))
MIN_RAG_SIMILARITY = float(os.getenv("MIN_RAG_SIMILARITY", "0.20"))

HF_HOME_LOCAL = os.getenv("HF_HOME_LOCAL", "").strip()
if HF_HOME_LOCAL:
    os.environ["HF_HOME"] = str(Path(HF_HOME_LOCAL).expanduser().resolve())

# -----------------------------------------------------------------------------
# 1. GLOBAL MODEL CACHE
# -----------------------------------------------------------------------------
_EMBEDDER = None
_EMBEDDER_ID = None

# -----------------------------------------------------------------------------
# 2. EXTERNAL HELPERS
# -----------------------------------------------------------------------------
from validator_utils import file_ext, normalize_space, weighted_average
from validator_llm import (DEFAULT_FALLBACK_LLM_MODEL_ID,DEFAULT_LLM_MODEL_ID,LLM_LOAD_MODE,cuda_summary,load_llm,unload_llm)
from validator_tasks import (DEFAULT_AUX_LLM_MODEL_ID,DEFAULT_JOB_SCHEMA_XLSX_PATH,extract_candidate_summary,extract_job_requirements,evaluate_one_requirement,load_job_requirement_schema_text,load_manual_job_requirements_from_excel)

# -----------------------------------------------------------------------------
# 3. DOCUMENT LOADING: PDF/DOCX/RTF/TXT/DOC
# -----------------------------------------------------------------------------

def load_pdf(path: str) -> str:
    fitz = import_module("fitz")  # PyMuPDF

    parts = []
    with fitz.open(path) as doc:
        page_count = int(getattr(doc, "page_count", 0))

        for page_index in range(page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""

            if text.strip():
                parts.append(f"\n--- PAGE {page_index + 1} ---\n{text}")

    return normalize_space("\n".join(parts))

def load_docx(path: str) -> str:
    import docx

    d = docx.Document(path)
    paragraphs = [p.text for p in d.paragraphs if p.text and p.text.strip()]

    for table in d.tables:
        for row in table.rows:
            cells = [normalize_space(c.text) for c in row.cells if c.text]
            if cells:
                paragraphs.append(" | ".join(cells))

    return normalize_space("\n".join(paragraphs))

def load_rtf(path: str) -> str:
    from striprtf.striprtf import rtf_to_text

    raw = Path(path).read_text(errors="ignore")
    return normalize_space(rtf_to_text(raw))

def load_txt(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1250", "latin-1"):
        try:
            return normalize_space(Path(path).read_text(encoding=enc, errors="ignore"))
        except Exception:
            continue
    return normalize_space(Path(path).read_text(errors="ignore"))

def load_doc_legacy_windows(path: str) -> str:
    try:
        import win32com.client  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Subor .doc je legacy format. Pre native Windows fallback treba mat nainstalovany "
            "Microsoft Word + pywin32. Alternativa: uloz CV ako .docx alebo .pdf."
        ) from exc

    tmp_dir = Path(tempfile.mkdtemp(prefix="cvdoc_"))
    tmp_txt = tmp_dir / "converted.txt"
    word = None
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(str(Path(path).resolve()))
        doc.SaveAs(str(tmp_txt), FileFormat=7)  # 7 = wdFormatUnicodeText
        doc.Close(False)
        return load_txt(str(tmp_txt))
    finally:
        try:
            if word is not None:
                word.Quit()
        except Exception:
            pass

def load_document(path: str) -> str:
    ext = file_ext(path)
    if ext == "pdf":
        return load_pdf(path)
    if ext == "docx":
        return load_docx(path)
    if ext == "rtf":
        return load_rtf(path)
    if ext in {"txt", "md"}:
        return load_txt(path)
    if ext == "doc":
        return load_doc_legacy_windows(path)
    raise ValueError(f"Nepodporovany format suboru: .{ext}. Pouzi PDF, DOCX, RTF, TXT alebo DOC.")

# -----------------------------------------------------------------------------
# 4. JOB AD SCRAPING
# -----------------------------------------------------------------------------

def scrape_url(url: str) -> str:
    if not url or not url.strip():
        return ""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
        )
    }
    resp = requests.get(url.strip(), headers=headers, timeout=25)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n")
    lines = [normalize_space(x) for x in text.splitlines()]
    lines = [x for x in lines if len(x) > 2]

    seen = set()
    unique = []
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(line)

    return normalize_space("\n".join(unique))

# -----------------------------------------------------------------------------
# 5. CHUNKING + RAG
# -----------------------------------------------------------------------------

def chunk_text(text: str, words_per_chunk: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, words_per_chunk - overlap)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + words_per_chunk]
        if len(chunk_words) < 20 and chunks:
            break
        chunks.append(" ".join(chunk_words))
    return chunks

def get_embedder(model_id: str = DEFAULT_EMBED_MODEL_ID):
    global _EMBEDDER, _EMBEDDER_ID
    if _EMBEDDER is not None and _EMBEDDER_ID == model_id:
        return _EMBEDDER
    device = "cpu"
    _EMBEDDER = SentenceTransformer(model_id, device=device)
    _EMBEDDER_ID = model_id
    return _EMBEDDER

def build_faiss_index(chunks: List[str], embed_model_id: str) -> Tuple[Any, Any]:
    if not chunks:
        raise ValueError("CV neobsahuje ziadny pouzitelny text/chunk.")

    embedder = get_embedder(embed_model_id)

    vectors = embedder.encode(
                                chunks,
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                                show_progress_bar=True,
                            )

    np_mod = cast(Any, np)
    vectors = np_mod.asarray(vectors, dtype="float32")

    if vectors.ndim != 2:
        raise ValueError(f"Embedding model vratil necakany tvar vektorov: {vectors.shape}")

    faiss_mod = cast(Any, faiss)
    index = cast(Any, faiss_mod.IndexFlatIP(int(vectors.shape[1])))
    add_fn = cast(Any, index.add)
    add_fn(vectors)

    return index, vectors

def rag_search(query: str, chunks: List[str], index: Any, embed_model_id: str, top_k: int) -> List[str]:
    embedder = get_embedder(embed_model_id)
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    scores, ids = index.search(q, min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        if float(score) < MIN_RAG_SIMILARITY:
            continue
        results.append(f"[similarity={float(score):.3f}] {chunks[int(idx)]}")
    return results

# -----------------------------------------------------------------------------
# 6. REPORTING
# -----------------------------------------------------------------------------

def status_icon(status: str) -> str:
    return {"splnene": "✅","ciastocne_splnene": "🟡","nesplnene": "❌","nejasne": "⚪",}.get(status, "⚪")

def verdict(score: float) -> str:
    if score >= 80:
        return "Vhodny kandidat"
    if score >= 60:
        return "Skor vhodny kandidat"
    if score >= 40:
        return "Ciastocne vhodny / vyzaduje manualne posudenie"
    return "Slaba zhoda s poziciou"

def render_markdown_report(job_data: Dict[str, Any], candidate: Dict[str, Any], evals: List[Dict[str, Any]]) -> str:
    score = weighted_average(evals)
    lines = []
    lines.append("# AI CV validator")
    lines.append("")
    lines.append(f"**Pozicia:** {job_data.get('job_title', 'unknown')}")
    lines.append(f"**Seniorita:** {job_data.get('seniority', 'unknown')}")
    lines.append(f"**Zdroj poziadaviek:** {job_data.get('_source', 'unknown')}")
    lines.append(f"**Celkove skore:** {score:.2f} / 100")
    lines.append(f"**Odporucanie:** {verdict(score)}")
    lines.append("")
    lines.append("> Vystup je odporucanie pre cloveka, nie automaticke rozhodnutie o kandidatovi.")
    lines.append("")

    if candidate:
        lines.append("## Anonymizovany profil kandidata")
        lines.append(str(candidate.get("summary", "")))
        for key, title in [
            ("skills", "Zrucnosti"),
            ("experience", "Skusenosti"),
            ("education", "Vzdelanie"),
            ("languages", "Jazyky"),
            ("certifications", "Certifikaty"),
            ("risks_or_missing_info", "Rizika / chybajuce info"),
        ]:
            vals = candidate.get(key)
            if isinstance(vals, list) and vals:
                lines.append(f"\n**{title}:**")
                for v in vals[:12]:
                    lines.append(f"- {v}")
        lines.append("")

    lines.append("## Vyhodnotenie poziadaviek")
    lines.append("")
    lines.append("| Stav | Poziadavka | Skore | Priorita | Vysvetlenie |")
    lines.append("|---|---|---:|---|---|")
    for r in evals:
        exp = str(r.get("explanation", "")).replace("|", "/")
        req = str(r.get("requirement", "")).replace("|", "/")
        lines.append(
            f"| {status_icon(r.get('status', ''))} {r.get('status', '')} "
            f"| {req} | {float(r.get('score', 0)):.0f} "
            f"| {r.get('priority', '')} / w={r.get('weight', 1)} | {exp} |"
        )

    lines.append("")
    lines.append("## Odkazy a poznamky")
    for r in evals:
        lines.append(f"\n### {status_icon(r.get('status', ''))} {r.get('requirement_id', '')}: {r.get('requirement', '')}")
        if r.get("risk_note"):
            lines.append(f"**Riziko/neistota:** {r.get('risk_note')}")
        ev = r.get("evidence_used") or []
        if isinstance(ev, list) and ev:
            lines.append("**Pouzite odkazy:**")
            for e in ev[:3]:
                short = normalize_space(str(e))[:700]
                lines.append(f"- {short}")

    return "\n".join(lines)

# -----------------------------------------------------------------------------
# 7. MAIN PIPELINE
# -----------------------------------------------------------------------------

def run_validation(
                    cv_file: str,
                    job_url: str,
                    job_text_manual: str,
                    job_schema_xlsx_path: str,
                    manual_position_name: str,
                    model_id: str,
                    fallback_model_id: str,
                    aux_model_id: str,
                    load_mode: str,
                    embed_model_id: str,
                    top_k: int,
                    max_requirements: int,
                    include_candidate_summary: bool,
) -> Tuple[str, str, str]:
    if not cv_file:
        raise gr.Error("Nahraj CV subor.")

    runtime = []
    runtime.append(cuda_summary())
    runtime.append(f"LLM model: {model_id}")
    runtime.append(f"Fallback model: {fallback_model_id}")
    runtime.append(f"Aux model: {aux_model_id or model_id}")
    runtime.append(f"Load mode: {load_mode}")
    runtime.append(f"Embedding model: {embed_model_id}")
    runtime.append(f"Top-K: {top_k}")
    runtime.append(f"Max requirements: {max_requirements}")
    runtime.append(f"Schema XLSX path: {job_schema_xlsx_path or DEFAULT_JOB_SCHEMA_XLSX_PATH}")
    runtime.append(f"Manual position: {manual_position_name or '-'}")
    runtime.append("Prompt mode: LangChain few-shot")

    cv_text = load_document(cv_file)
    if len(cv_text) < 100:
        raise gr.Error("Z CV sa podarilo vytiahnut velmi malo textu. Skus iny format, idealne PDF/DOCX.")
    runtime.append(f"CV text: {len(cv_text):,} znakov")

    job_requirement_schema_text, prompt_schema_source = load_job_requirement_schema_text(
        job_schema_xlsx_path or DEFAULT_JOB_SCHEMA_XLSX_PATH
    )
    runtime.append(f"Prompt schema source: {prompt_schema_source}")

    _, _, model_info = load_llm(model_id, load_mode, fallback_model_id)
    runtime.append(model_info)

    if manual_position_name and manual_position_name.strip():
        job_data = load_manual_job_requirements_from_excel(
                                                            position_query=manual_position_name,
                                                            schema_xlsx_path=job_schema_xlsx_path or DEFAULT_JOB_SCHEMA_XLSX_PATH,
                                                            max_requirements=max_requirements,
                                                            model_id=model_id,
                                                            load_mode=load_mode,
                                                            fallback_model_id=fallback_model_id,
                                                            aux_model_id=aux_model_id,
                                                            )
        runtime.append("Zdroj pozicie: manualny schema XLSX katalog")
    else:
        job_text = ""
        if job_text_manual and job_text_manual.strip():
            job_text = normalize_space(job_text_manual)
            runtime.append("Inzerat: pouzity manualne vlozeny text")
        elif job_url and job_url.strip():
            job_text = scrape_url(job_url)
            runtime.append(f"Inzerat: nacitany z URL, {len(job_text):,} znakov")
        else:
            raise gr.Error("Zadaj URL inzeratu, vloz text inzeratu manualne, alebo vypln manualnu poziciu zo schema XLSX.")

        if len(job_text) < 100:
            raise gr.Error("Z inzeratu sa podarilo ziskat velmi malo textu. Vloz text inzeratu manualne.")

        job_data = extract_job_requirements(
                                            job_text,
                                            job_requirement_schema_text,
                                            model_id,
                                            load_mode,
                                            fallback_model_id,
                                            aux_model_id,
                                            max_requirements,
                                            )
    requirements = job_data.get("requirements", [])
    if not requirements:
        raise gr.Error("LLM ani fallback neextrahovali ziadne poziadavky z inzeratu. Skus vlozit cistejsi text inzeratu.")
    runtime.append(f"Extrahovane poziadavky: {len(requirements)}")
    runtime.append(f"Zdroj poziadaviek: {job_data.get('_source', 'unknown')}")

    meta = job_data.get("_meta", {})
    if isinstance(meta, dict):
        runtime.append(f"LLM count: {meta.get('llm_count', 0)}")
        runtime.append(f"Fallback count: {meta.get('fallback_count', 0)}")
        runtime.append(f"Weak LLM: {meta.get('weak_llm', False)}")
        runtime.append(f"Merged count: {meta.get('merged_count', 0)}")
        runtime.append(f"Prompt mode meta: {meta.get('prompt_mode', 'few_shot_langchain')}")

    candidate = {}
    if include_candidate_summary:
        candidate = extract_candidate_summary(cv_text, model_id, load_mode, fallback_model_id)

    chunks = chunk_text(cv_text, CHUNK_WORDS, CHUNK_OVERLAP)
    index, _ = build_faiss_index(chunks, embed_model_id)
    runtime.append(f"CV chunks: {len(chunks)}")

    evals = []
    for req in requirements:
        evidence = rag_search(req.get("text", ""), chunks, index, embed_model_id, int(top_k))
        ev = evaluate_one_requirement(req, evidence, model_id, load_mode, fallback_model_id)
        evals.append(ev)

    final = {
                "job": job_data,
                "candidate_profile": candidate,
                "overall_score": weighted_average(evals),
                "verdict": verdict(weighted_average(evals)),
                "evaluations": evals,
                "runtime": runtime,
            }

    md = render_markdown_report(job_data, candidate, evals)
    js = json.dumps(final, ensure_ascii=False, indent=2)
    return md, js, "\n".join(runtime)

def gradio_run_wrapper(*args):
    try:
        return run_validation(*args)
    except gr.Error:
        raise
    except Exception as exc:
        err = f"Chyba: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        return "# Chyba pri spracovani\n\n```text\n" + err + "\n```", "{}", err

# -----------------------------------------------------------------------------
# 8. GRADIO UI
# -----------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Lokalny AI CV Validator") as demo:
        gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=1):
                cv_file = gr.File(
                    label="CV subor",
                    file_types=[".pdf", ".docx", ".doc", ".rtf", ".txt", ".md"],
                    type="filepath",
                )
                job_url = gr.Textbox(label="URL inzeratu", placeholder="https://")
                job_text_manual = gr.Textbox(
                    label="Alebo vloz text inzeratu manualne",
                    lines=8,
                    placeholder="Text pracovnej ponuky",
                )

                with gr.Accordion("Externy schema XLSX", open=False):
                    job_schema_xlsx_path = gr.Textbox(
                        label="Schema XLSX path",
                        value=DEFAULT_JOB_SCHEMA_XLSX_PATH,
                    )
                    manual_position_name = gr.Textbox(
                        label="Manualna pozicia zo schema XLSX",
                        placeholder="napr. python_backend_medior alebo Python developer",
                    )

                with gr.Accordion("Model nastavenia", open=False):
                    model_id = gr.Textbox(label="LLM model z Hugging Face", value=DEFAULT_LLM_MODEL_ID)
                    fallback_model_id = gr.Textbox(label="Fallback LLM model", value=DEFAULT_FALLBACK_LLM_MODEL_ID)
                    aux_model_id = gr.Textbox(label="Aux LLM model pre canonicalizaciu/genericnost", value=DEFAULT_AUX_LLM_MODEL_ID or DEFAULT_LLM_MODEL_ID)
                    load_mode = gr.Dropdown(
                        label="Load mode",
                        choices=["auto", "bnb_4bit", "fp16_gpu", "cpu"],
                        value=LLM_LOAD_MODE,
                    )
                    embed_model_id = gr.Textbox(label="Embedding model z Hugging Face", value=DEFAULT_EMBED_MODEL_ID)
                    top_k = gr.Slider(label="Top-K dokazov z CV", minimum=2, maximum=20, step=1, value=DEFAULT_TOP_K)
                    max_requirements = gr.Slider(
                        label="Max pocet poziadaviek z inzeratu",
                        minimum=5,
                        maximum=25,
                        step=1,
                        value=DEFAULT_MAX_REQUIREMENTS,
                    )
                    include_candidate_summary = gr.Checkbox(label="Extrahovat anonymizovany profil kandidata", value=True)

                with gr.Row():
                    run_btn = gr.Button("Spustit validaciu", variant="primary")
                    unload_btn = gr.Button("Uvolnit model z VRAM")

            with gr.Column(scale=2):
                report_md = gr.Markdown(label="Report")
                runtime_info = gr.Textbox(label="Runtime info", lines=8)
                json_report = gr.Code(label="JSON report", language="json", lines=20)

        run_btn.click(
            fn=gradio_run_wrapper,
            inputs=[
                    cv_file,
                    job_url,
                    job_text_manual,
                    job_schema_xlsx_path,
                    manual_position_name,
                    model_id,
                    fallback_model_id,
                    aux_model_id,
                    load_mode,
                    embed_model_id,
                    top_k,
                    max_requirements,
                    include_candidate_summary,
                    ],
            outputs=[report_md, json_report, runtime_info],
        )
        unload_btn.click(fn=unload_llm, inputs=[], outputs=[runtime_info])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)