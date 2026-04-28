# AI CV Validator 

## 1. Konfiguracia

Odporucane pre RTX 4070 Super 12GB:

```env

LLM_MODEL_ID=Qwen/Qwen3-4B-Thinking-2507
FALLBACK_LLM_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
EMBED_MODEL_ID=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
LLM_LOAD_MODE=auto
MAX_GPU_MEMORY=10.5GiB

```

## 2. Ako funguje pipeline

```text

CV + URL/text inzeratu (Gradio frontend)
        |
        v
Extrakcia textu
        |
        v
LLM extrahuje poziadavky z inzeratu
        |
        v
CV sa rozdeli na chunky
        |
        v
Embedding model vytvori FAISS index
        |
        v
Ku kazdej poziadavke sa najdu dokazy z CV
        |
        v
LLM vyhodnoti poziadavku z odkazov
        |
        v
Report + JSON vystup

```

## 3. Podporovane dokumenty
```text

• PDF cez PyMuPDF
• DOCX cez python-docx
• RTF cez striprtf
• TXT/MD priamo
• DOC legacy cez Microsoft Word COM fallback, ak je nainstalovany Word + pywin32

```