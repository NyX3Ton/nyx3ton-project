# HR Attrition tool (backend + frontend)

## 1. Konfiguracia
```text
Filename: process_legal.py

a. Nastavenie postu vlakien pre CPU
torch.set_num_threads(24)

b. Nastavenie fragmentacie pamate GPU
USE_CUDA = torch.cuda.is_available()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                                        "expandable_segments:True,"
                                        "max_split_size_mb:128,"
                                        "garbage_collection_threshold:0.8"
                                        )
c. Windows Native odporucana konfiguracia
if USE_CUDA:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

d. Pouzite modely
main_model_id = nvidia/Nemotron-Terminal-8B (zalozny Qwen/Qwen3-4B-Thinking-2507)
summ_model_id = slovak-nlp/mistral-sk-7b
emb_model_id = sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

## 2. Ako funguje pipeline

```text
- podporovane formaty (DOCX (cez Docx2txtLoader), TXT (cez TextLoader a encoding utf-8), RTF striprtf))

Dokumenty konvertovane do pandas dataframe
        |
        v
Nacitanie Stopwords vo formate TXT
        |
        v
Nacitanie Thesauru vo formate CSV
        |
        v
CPU model pre klasifikaciu viet
        |
        v
CPU summarizacny model
        |
        v
GPU model pre generovanie vysledkov (keyword alebo title)
        |
        v
Ulozenie vysledkov do CSV formatu
```

## 3. Obsah
```text
1. CONFIGURATION
2. HELPERS
  a. loading stopwords
  b. loading Thesaurus
  c. verify/validate quantization before model load
  d. model load from previous model save, if fails continue from scratch
  e. model unload (explicit RAM/VRAM release)
3. RAG MANAGER
4. PIPELINE  — three sequential phases, one model resident at a time
  a. Phase 1 – Ingest: read all files, build RAG (sentence-transformer only)
  b. Phase 2 – Summarise: CPU summariser → summarise all → unload
  c. Phase 3 – Generate: CUDA generator → RAG-retrieve + generate → unload
5. MAIN FUNCTION
```
