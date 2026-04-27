# Lokalny AI CV Validator 

## 1. Konfiguracia

Odporucane pre RTX 4070 Super 12GB:

```env
LLM_MODEL_ID=nvidia/Llama-3.1-Nemotron-Nano-8B-v1
FALLBACK_LLM_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
EMBED_MODEL_ID=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
LLM_LOAD_MODE=auto
MAX_GPU_MEMORY=10.5GiB
```

## 2. Ako funguje pipeline

```text
CV + URL/text inzeratu
        |
        v
extrakcia textu
        |
        v
LLM extrahuje poziadavky z inzeratu
        |
        v
CV sa rozdeli na chunky
        |
        v
embedding model vytvori FAISS index
        |
        v
ku kazdej poziadavke sa najdu dokazy z CV
        |
        v
LLM vyhodnoti poziadavku iba z dokazov
        |
        v
report + JSON vystup
```

## 3. Podporovane dokumenty

- PDF cez PyMuPDF
- DOCX cez python-docx
- RTF cez striprtf
- TXT/MD priamo
- DOC legacy cez Microsoft Word COM fallback, ak je nainstalovany Word + pywin32
