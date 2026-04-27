# Lokalny AI CV Validator - single-file Windows MVP

Tato verzia je zjednodusena na minimum Python suborov:

```text
app.py              # cela aplikacia: loader, scraper, RAG, LLM, Gradio UI
requirements.txt    # kniznice
.env.example        # konfiguracia
run_app_windows.bat # rychle spustenie
```

Modely sa nestahuju manualne. Pri prvom spusteni ich automaticky stiahne Hugging Face cache cez `transformers` a `sentence-transformers`.

## 1. Instalacia vo VS Code bez virtualneho prostredia

Otvor priecinok vo VS Code a v terminali spusti:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Ak PyTorch nenajde CUDA, preinstaluj GPU build:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Potom over:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## 2. Konfiguracia

```powershell
copy .env.example .env
```

Odporucane pre RTX 4070 Super 12GB:

```env
LLM_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
FALLBACK_LLM_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
LLM_LOAD_MODE=auto
MAX_GPU_MEMORY=10.5GiB
```

## 3. Spustenie

```powershell
python app.py
```

Alebo dvojklik/spustenie:

```text
run_app_windows.bat
```

UI bude na:

```text
http://127.0.0.1:7860
```

## 4. Podporovane dokumenty

- PDF cez PyMuPDF
- DOCX cez python-docx
- RTF cez striprtf
- TXT/MD priamo
- DOC legacy cez Microsoft Word COM fallback, ak je nainstalovany Word + pywin32

Pre najlepsiu spolahlivost odporucam CV vo formate PDF alebo DOCX.

## 5. Ako funguje pipeline

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

## 6. Poznamky k Windows/bitsandbytes

`LLM_LOAD_MODE=auto` je najpraktickejsi:

1. skusi 7B model v 4-bit kvantizacii,
2. ak zlyha, skusi 3B model vo FP16 na GPU,
3. ak nie je CUDA, skusi 3B na CPU.

Pre bakalarsku pracu je toto vyhodne, lebo vies zdokumentovat lokalny deployment, ochranu dat a fallback strategiu.
