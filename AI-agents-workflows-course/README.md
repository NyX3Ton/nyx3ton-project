# AI prompt simulator 

```text
https://gitlab.com/nyx3ton-group/nyx3ton-project/-/blob/main/AI-agents-workflows-course/ai-prompt-sim.py

```
- Upravene na beh v Docker-compose
- Architektura primarne bezi na OpenVINO (boal pridana detekcia pre CUDA).
- Podporuje upravy MAX_INPUT_TOKENS, TOP_P a TEMPERATURE v Gradio UI
- Podporuje Zero-shot, Few-Shot, User and System Prompt
- Podporuje viacpolozkovy Web Scrapping cez BeautifulSoup (non-JavaScript Web URLs)
- Architektura podporuje a je merana pomocou MLFlow
- Poddporuje faiss-cpu ako primarny RAG (podpora pre .txt, .md, .pdf, .docx, .rtf)
- RAG podporuje konfiguraciu pre Chunk size, Top-K a Chunk-overlap
- podporuje zapis do Markdown (aj s metadatami)


```text
Lokalne modely pre ktore bol skript testovany su:

Qwen/Qwen3-0.6B
https://huggingface.co/Qwen/Qwen3-0.6B

Qwen/Qwen3-4B-Instruct-2507
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

Sentence trasformer pre RAG:
https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2

```

![Qwen3-0.6B](images/Qwen3.png)

Povinne kniznice:

```text
!python -m pip install --upgrade pip setuptools wheel
!python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
!python -m pip install --upgrade ipykernel jupyter
!python -m pip install --upgrade mlflow
!python -m pip install --upgrade openvino optimum-intel transformers accelerate safetensors sentencepiece huggingface_hub requests python-dotenv gradio bs4
!python -m pip install --upgrade hf_xet nncf
!python -m pip install --upgrade sentence-transformers faiss-cpu numpy python-docx PyMuPDF striprtf
```
Spustenie v dockeri:
```text
docker compose -f docker-compose.cpu.yml build --no-cache --progress=plain
docker compose -f docker-compose.cpu.yml up --force-recreate
```

Premazanie v Dockeri
```text
docker compose -f docker-compose.cpu.yml down --remove-orphans
docker rm -f ai-prompt-sim-rag-cpu 2>$null
docker image rm ai-prompt-sim-rag:cpu 2>$null
```

## 1. Konfiguracia

Odporucane pre x86-64 Processory s porpodou OpenVINO (AVX-512):

```env

Priecinky pre lokalne behy modelov:

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "./hf_cache")
OV_MODELS_DIR = os.getenv("OV_MODELS_DIR", "./ov_models")
OV_CACHE_DIR = os.getenv("OV_CACHE_DIR", "./ov_cache")

Nastavenie zariadenia:

OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "CPU")

Vychodiskoive nastavenia modelu (mozna zmena z Gradio UI)

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "2048"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

Vynutenie behu modelku z lokalneho priecinku:

OFFLINE_MODE = os.getenv("LOCAL_FILES_ONLY", "1").strip().lower() in {"1", "true", "yes", "y"}
FORCE_HF_FALLBACK = os.getenv("FORCE_HF_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "y"}

```
### 2. Vystup z MLFlow a porovnanie behov


![MLFlow](images/ml_flow_incorporation.png)

### 3. Priklad vystupu z modelu

Menu s outputom:

![Dashboard](images/asistant-output.png)


