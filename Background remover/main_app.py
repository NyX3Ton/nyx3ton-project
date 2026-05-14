# filename: main_app.ipynb

!pip install gradio rembg pillow onnxruntime-gpu
# -----------------------------------------------------------------------------
# 0. IMPORTS
# 1. HELPERS
# 2. ONNX RUNTIME
# 3. FILES\OUTPUTS\FORMATS
# 4. MAIN LOGIC
# 5. GRADIO UI
# -----------------------------------------------------------------------------

# --------------------------------------------------
# 0. IMPORTS
# --------------------------------------------------

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import onnxruntime as ort
from PIL import Image, ImageOps
from rembg import remove, new_session

OUTPUT_DIR = Path("output_images")
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# --------------------------------------------------
# 1. HELPERS 
# --------------------------------------------------
OUTPUT_DIR = Path("output_images")
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MODEL_OPTIONS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
]

PROVIDER_MODE_OPTIONS = [
    "CUDA / GPU",
    "TensorRT / experimental",
    "CPU",
]

SESSION_CACHE: Dict[Tuple[str, str], Tuple[Any, str, List[str]]] = {}

# --------------------------------------------------
# 2. ONNX RUNTIME 
# --------------------------------------------------
def get_available_runtime_info() -> str:
    providers = ort.get_available_providers()

    lines = ["Dostupne ONNX Runtime providery:", "", ]

    for provider in providers:
        lines.append(f"- {provider}")

    lines.append("")

    if "CUDAExecutionProvider" in providers:
        lines.append("Stav GPU: CUDAExecutionProvider je dostupny.")
    else:
        lines.append("Stav GPU: CUDAExecutionProvider nie je dostupny, pouzije sa CPU.")

    if "TensorrtExecutionProvider" in providers:
        lines.append("TensorRT: dostupny, ale odporucam ho brat ako experimental.")
    else:
        lines.append("TensorRT: nie je dostupny.")

    return "\n".join(lines)

def build_provider_candidates(provider_mode: str) -> List[Tuple[str, List[str]]]:
    available = ort.get_available_providers()
    candidates: List[Tuple[str, List[str]]] = []

    if provider_mode == "TensorRT / experimental":
        if "TensorrtExecutionProvider" in available:
            candidates.append(("TensorRT / experimental",["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"],))

        if "CUDAExecutionProvider" in available:
            candidates.append(("CUDA / GPU fallback",["CUDAExecutionProvider","CPUExecutionProvider"]))

        candidates.append(("CPU fallback",["CPUExecutionProvider"]))

    elif provider_mode == "CUDA / GPU":
        if "CUDAExecutionProvider" in available:
            candidates.append(("CUDA / GPU",["CUDAExecutionProvider","CPUExecutionProvider",]))

        candidates.append(
            ("CPU fallback",["CPUExecutionProvider"]))

    else:
        candidates.append(("CPU",["CPUExecutionProvider",]))

    return candidates

def get_active_providers_from_session(session: Any) -> List[str]:
    try:
        return list(session.inner_session.get_providers())
    except Exception:
        pass

    try:
        return list(session.session.get_providers())
    except Exception:
        pass

    try:
        return list(session.get_providers())
    except Exception:
        pass

    return ["Neviem nacitat aktivne providery zo session."]

def get_session(model_name: str, provider_mode: str):
    cache_key = (model_name, provider_mode)

    if cache_key in SESSION_CACHE:
        return SESSION_CACHE[cache_key]

    errors = []
    candidates = build_provider_candidates(provider_mode)

    for provider_label, providers in candidates:
        try:
            session = new_session(model_name,providers=providers,)

            active_providers = get_active_providers_from_session(session)
            SESSION_CACHE[cache_key] = (session, provider_label, active_providers)

            return SESSION_CACHE[cache_key]

        except Exception as exc:
            errors.append(f"{provider_label}: {str(exc)}")

    raise RuntimeError("Nepodarilo sa vytvorit rembg session ani cez fallback.\n" + "\n".join(errors))

# --------------------------------------------------
# 3. FILES\OUTPUTS\FORMATS
# --------------------------------------------------
def sanitize_suffix(suffix: str) -> str:
    suffix = suffix.strip()

    if not suffix:
        suffix = "_bez_pozadia"

    if not suffix.startswith("_"):
        suffix = "_" + suffix

    suffix = re.sub(r'[<>:"/\\|?*]', "_", suffix)
    suffix = re.sub(r"\s+", "_", suffix)

    return suffix

def get_unique_output_path(input_path: Path, suffix: str) -> Path:
    ext = input_path.suffix.lower()
    stem = input_path.stem

    suffix = sanitize_suffix(suffix)

    output_path = OUTPUT_DIR / f"{stem}{suffix}{ext}"
    counter = 1

    while output_path.exists():
        output_path = OUTPUT_DIR / f"{stem}{suffix}_{counter}{ext}"
        counter += 1

    return output_path

def hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    try:
        value = hex_color.strip().replace("#", "")

        if len(value) != 6:
            return 255, 255, 255, alpha

        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)

        return r, g, b, alpha

    except Exception:
        return 255, 255, 255, alpha

def save_same_format(output_image: Image.Image,output_path: Path,ext: str,jpg_background_color: str,jpg_quality: int):
    output_image = output_image.convert("RGBA")

    if ext == ".png":
        output_image.save(output_path, format="PNG")
        return

    if ext in {".jpg", ".jpeg"}:
        bg_rgba = hex_to_rgba(jpg_background_color)
        background = Image.new("RGBA", output_image.size, bg_rgba)
        background.alpha_composite(output_image)

        rgb_image = background.convert("RGB")
        rgb_image.save(output_path,format="JPEG",quality=int(jpg_quality),optimize=True,subsampling=0)
        return

    raise ValueError(f"Nepodporovany vystupny format: {ext}")

# --------------------------------------------------
# 4. MAIN LOGIC
# --------------------------------------------------

def remove_background_and_save(
    image_file,
    model_name: str,
    provider_mode: str,
    alpha_matting: bool,
    alpha_fg_threshold: int,
    alpha_bg_threshold: int,
    alpha_erode_size: int,
    post_process_mask: bool,
    only_mask: bool,
    jpg_background_color: str,
    jpg_quality: int,
    output_suffix: str):

    if image_file is None:
        return None, None, "Nebol nahraty ziadny subor."

    input_path = Path(image_file)
    ext = input_path.suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return None, None, "Nepodporovany format. Pouzi JPG, JPEG alebo PNG."

    try:
        input_image = Image.open(input_path)
        input_image = ImageOps.exif_transpose(input_image)
        input_image = input_image.convert("RGBA")

        session, active_provider_label, active_providers = get_session(model_name=model_name,provider_mode=provider_mode)

        output_image = remove(
                                input_image,
                                session=session,
                                alpha_matting=alpha_matting,
                                alpha_matting_foreground_threshold=int(alpha_fg_threshold),
                                alpha_matting_background_threshold=int(alpha_bg_threshold),
                                alpha_matting_erode_size=int(alpha_erode_size),
                                post_process_mask=post_process_mask,
                                only_mask=only_mask
                            )

        if not isinstance(output_image, Image.Image):
            raise RuntimeError("Neocakavany vystup z rembg. Očakaval sa PIL Image.")

        output_image = output_image.convert("RGBA")

        output_path = get_unique_output_path(input_path=input_path,suffix=output_suffix)

        save_same_format(
                            output_image=output_image,
                            output_path=output_path,
                            ext=ext,
                            jpg_background_color=jpg_background_color,
                            jpg_quality=int(jpg_quality),
                        )

        status = (
                    "Hotovo.\n\n"
                    f"Vstupny subor: {input_path.name}\n"
                    f"Vystupny subor: {output_path.name}\n"
                    f"Model: {model_name}\n"
                    f"Zvoleny backend: {provider_mode}\n"
                    f"Realne pouzity backend: {active_provider_label}\n"
                    f"Aktivne providery: {active_providers}\n"
                    f"Alpha matting: {alpha_matting}\n"
                    f"Post-processing masky: {post_process_mask}\n"
                    f"Only mask: {only_mask}"
                )

        return str(output_path), str(output_path), status

    except Exception as exc:
        return None, None, f"Nastala chyba:\n{str(exc)}"

def apply_preset(preset_name: str):
    if preset_name == "Produktova fotka - kvalita":
        return ("isnet-general-use",True,240,10,10,True,False)

    if preset_name == "Portret / osoba":
        return ("u2net_human_seg",True,240,10,12,True,False)

    if preset_name == "Rychle spracovanie":
        return ("u2netp",False,240,10,10,True,False)

    if preset_name == "Jemne okraje / vlasy":
        return ("isnet-general-use",True,220,20,5,True,False)

    if preset_name == "Diagnostika masky":
        return ("isnet-general-use",False,240,10,10,True,True)

    return ("isnet-general-use",True,240,10,10,True,False)


def clear_all():
    return None, None, None, ""

# --------------------------------------------------
# 5. GRADIO UI
# --------------------------------------------------

with gr.Blocks(title="AI odstranenie pozadia",theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI odstranenie pozadia z obrazka")
    gr.Markdown(
                "Nahraj obrazok vo formate **JPG, JPEG alebo PNG**. "
                "Aplikacia odstrani pozadie pomocou neuronovej siete, pouzije GPU ak je dostupne "
                "a ulozi vysledok v rovnakom formate s upravenym nazvom suboru."
                )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.File(label="Vstupny obrazok",file_types=[".jpg", ".jpeg", ".png"],type="filepath")

            process_btn = gr.Button("Odstranit pozadie",variant="primary")

            clear_btn = gr.Button("Vymazat")

        with gr.Column(scale=1):
            output_preview = gr.Image(label="Nahlad vysledku",type="filepath")

            output_file = gr.File(label="Stiahnut vysledny subor")

            status_text = gr.Textbox(label="Stav",interactive=False,lines=10)

    with gr.Accordion("GPU / ONNX Runtime diagnostika", open=True):
        runtime_info = gr.Textbox(value=get_available_runtime_info(),label="Dostupne backendy",interactive=False,lines=8)

        provider_mode = gr.Dropdown(choices=PROVIDER_MODE_OPTIONS,value="CUDA / GPU",label="Vypoctovy backend")

    with gr.Accordion("Pokrocile ladenie neuronovej siete", open=False):
        preset_name = gr.Dropdown(
            choices=[
                        "Produktova fotka - kvalita",
                        "Portret / osoba",
                        "Rychle spracovanie",
                        "Jemne okraje / vlasy",
                        "Diagnostika masky",
                        "Vlastne nastavenie",
                    ],value="Produktova fotka - kvalita",label="Rychly profil")

        model_name = gr.Dropdown(choices=MODEL_OPTIONS,value="isnet-general-use",label="Model neuronovej siete")

        output_suffix = gr.Textbox(value="_bez_pozadia",label="Suffix nazvu suboru",placeholder="_bez_pozadia")

        with gr.Row():
            alpha_matting = gr.Checkbox(value=True,label="Alpha matting - lepsie okraje objektu")
            post_process_mask = gr.Checkbox(value=True,label="Post-processing masky")
            only_mask = gr.Checkbox(value=False,label="Diagnosticky vystup iba ako maska")

        with gr.Row():
            alpha_fg_threshold = gr.Slider(minimum=0,maximum=300,value=240,step=1,label="Foreground threshold")
            alpha_bg_threshold = gr.Slider(minimum=0,maximum=300,value=10,step=1,label="Background threshold")
            alpha_erode_size = gr.Slider(minimum=0,maximum=50,value=10,step=1,label="Erode size")

        with gr.Row():
            jpg_background_color = gr.ColorPicker(value="#ffffff",label="Pozadie pre JPG/JPEG")
            jpg_quality = gr.Slider(minimum=50,maximum=100,value=95,step=1,label="JPEG kvalita")

        gr.Markdown(
                    """
                    ### Prakticke odporucanie

                    - **Produktova fotka - kvalita**: najlepsi default pre predmety, produkty, bezne fotky.
                    - **Portret / osoba**: vhodne pre ludi a portrety.
                    - **Rychle spracovanie**: mensi model, rychlejsi vysledok.
                    - **Jemne okraje / vlasy**: skus pri vlasoch, srsti, jemnych hranach.
                    - **Diagnostika masky**: zobrazi iba masku, vhodne na hladanie problemu.
                    """
                    )

    preset_name.change(
                        fn=apply_preset,
                        inputs=[preset_name],
                        outputs=[
                                model_name,
                                alpha_matting,
                                alpha_fg_threshold,
                                alpha_bg_threshold,
                                alpha_erode_size,
                                post_process_mask,
                                only_mask
                                ])

    process_btn.click(
                        fn=remove_background_and_save,
                        inputs=[
                                image_input,
                                model_name,
                                provider_mode,
                                alpha_matting,
                                alpha_fg_threshold,
                                alpha_bg_threshold,
                                alpha_erode_size,
                                post_process_mask,
                                only_mask,
                                jpg_background_color,
                                jpg_quality,
                                output_suffix,
                                ],
                        outputs=[
                                output_preview,
                                output_file,
                                status_text,
                                ])

    clear_btn.click(
                    fn=clear_all,
                    inputs=[],
                    outputs=[
                            image_input,
                            output_preview,
                            output_file,
                            status_text,
                            ])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
