import os
import threading
import time
import random
from pathlib import Path
from typing import Callable
from io import BytesIO

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from PIL import Image

from dotenv import load_dotenv
from google import genai


# -----------------------------
# Prompts
# -----------------------------
PROMPT_RESTORE_RECREATE = (
    "Can you recreate this image as close as possible, just upscale the image and try to preserve the faces. "
    "Keep the same people, identity, expressions, composition, and background. Do not add or remove anything. "
    "Do not stylize. Avoid over-smoothing. Keep it realistic."
)

PROMPT_UPSCALE_STRICT_BW = (
    "Upscale this photo to a higher resolution and improve clarity while preserving the original image exactly. "
    "Keep the same people, faces, expressions, clothing, background, and composition. "
    "Do not add or remove anything. Do not change identity. "
    "Reduce noise slightly, keep natural film grain, and avoid over-smoothing. "
    "Black and white only (no colorization)."
)

PROMPT_UPSCALE_STRICT_COLOR = (
    "Upscale this photo to a higher resolution and improve clarity while preserving the original image exactly. "
    "Keep the same colors and lighting (no re-coloring or re-stylizing). "
    "Do not add or remove anything. Do not change identity. "
    "Reduce noise slightly and avoid over-smoothing."
)

PROMPT_MODERN_2025 = (
    "Modernize this photo to look like a high-resolution photo from 2025. "
    "Improve clarity, dynamic range, and natural sharpness while keeping the same people, faces, expressions, "
    "composition, and background. Do not change identity. Do not add or remove objects. "
    "Keep it realistic, avoid plastic skin, and preserve a natural photo look (no stylized filters)."
)

# Gemini image edit model
MODEL_IMAGE = "gemini-2.5-flash-image"
MODEL_IMAGE_UPSCALE = "imagen-3.0-generate-001"


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# -----------------------------
# Helpers
# -----------------------------
def is_probably_bw(img: Image.Image) -> bool:
    """Simple heuristic: if RGB channels are very similar, treat as B/W."""
    img = img.convert("RGB").resize((96, 96))
    pixels = list(img.getdata())
    diffs = []
    for r, g, b in pixels:
        diffs.append(abs(r - g) + abs(g - b) + abs(r - b))
    return (sum(diffs) / len(diffs)) < 10


def ensure_output_folder(input_path: Path, chosen_output: str | None) -> Path:
    if chosen_output:
        out_dir = Path(chosen_output)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    base = input_path if input_path.is_dir() else input_path.parent
    out_dir = base / "upscaled"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = []
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    files.sort()
    return files


def _prompt_feedback_summary(response) -> str | None:
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is None:
        return None
    reason = getattr(feedback, "block_reason", None)
    message = getattr(feedback, "block_reason_message", None)
    parts = []
    if reason:
        parts.append(f"block_reason={reason}")
    if message:
        parts.append(f"message={message}")
    return ", ".join(parts) if parts else "blocked by safety filters"


def extract_first_image_from_response(response) -> Image.Image:
    if not getattr(response, "candidates", None):
        feedback = _prompt_feedback_summary(response)
        if feedback:
            raise RuntimeError(f"Model returned no candidates ({feedback}).")
        raise RuntimeError("Model returned no candidates.")
    content = response.candidates[0].content
    if content is None or not getattr(content, "parts", None):
        raise RuntimeError(
            "Model returned no image content (possibly blocked or empty response)."
        )
    for part in content.parts:
        if getattr(part, "inline_data", None) is not None:
            return Image.open(BytesIO(part.inline_data.data)).convert("RGB")
    raise RuntimeError("Model returned no image (text-only or blocked response).")


def gemini_edit_image(
    client: genai.Client,
    prompt: str,
    img: Image.Image,
    log: Callable[[str], None] | None = None,
    max_retries: int = 3,
) -> Image.Image:
    # Force image output (otherwise the model may respond with text only)
    cfg = genai.types.GenerateContentConfig(response_modalities=["IMAGE"])
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_IMAGE,
                contents=[prompt, img],
                config=cfg,
            )
            return extract_first_image_from_response(response)
        except Exception as e:
            last_error = e
            # If prompt feedback indicates safety block, don't retry.
            if "block_reason" in str(e).lower():
                break
            if attempt < max_retries and log is not None:
                wait_s = 0.6 * attempt + random.random() * 0.4
                log(f"     Retry {attempt}/{max_retries} after error: {e}")
                time.sleep(wait_s)
    raise RuntimeError(str(last_error) if last_error else "Gemini edit failed.")


def pil_to_genai_image(img: Image.Image, mime_type: str = "image/png") -> genai.types.Image:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return genai.types.Image(image_bytes=buf.getvalue(), mime_type=mime_type)


def extract_pil_from_genai_image(genai_image: genai.types.Image) -> Image.Image:
    if genai_image.image_bytes is None:
        raise RuntimeError("Upscale response contained no image bytes.")
    return Image.open(BytesIO(genai_image.image_bytes)).convert("RGB")


def vertex_upscale_image(
    client: genai.Client, img: Image.Image, upscale_factor: str
) -> Image.Image:
    response = client.models.upscale_image(
        model=MODEL_IMAGE_UPSCALE,
        image=pil_to_genai_image(img),
        upscale_factor=upscale_factor,
    )
    return extract_pil_from_genai_image(response.generated_images[0].image)


def build_upscale_prompt(mode: str, img: Image.Image) -> str:
    """
    mode in {"Mixed", "Black & white", "Color"}.
    If Mixed: detect from image.
    """
    if mode in {"Black & white", "Schwarz-weiß"}:
        return PROMPT_UPSCALE_STRICT_BW
    if mode in {"Color", "Farbe beibehalten"}:
        return PROMPT_UPSCALE_STRICT_COLOR
    return PROMPT_UPSCALE_STRICT_BW if is_probably_bw(img) else PROMPT_UPSCALE_STRICT_COLOR


def parse_scale_factor(value: str) -> int:
    if value in {"2x", "2× größer", "2× larger"}:
        return 2
    if value in {"4x", "4× größer", "4× larger"}:
        return 4
    return 2


def safe_sleep(seconds: float, app) -> None:
    """Sleep in small increments so Stop feels responsive."""
    end = time.time() + seconds
    while time.time() < end:
        if not app.is_running:
            return
        time.sleep(0.05)


class Tooltip:
    def __init__(self, widget, text: str, delay_ms: int = 600):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip_window = None

        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def update_text(self, text: str):
        self.text = text

    def _schedule(self, _event=None):
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self):
        if self._tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self._tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#111827",
            foreground="#f3f4f6",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 10),
            padx=12,
            pady=8,
        )
        label.pack()

    def _hide(self, _event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None


# -----------------------------
# GUI App
# -----------------------------
UI_TEXT = {
    "en": {
        "title": "Photo Restore + Upscale (Gemini)",
        "settings_label": "Settings",
        "menu_language": "Language",
        "menu_theme": "Appearance",
        "menu_lang_de": "Deutsch",
        "menu_lang_en": "English",
        "input_section": "2) Select Input (single image or folder)",
        "choose_image": "Choose Image…",
        "choose_folder": "Choose Folder…",
        "output_section": "3) Select Output Folder (optional)",
        "choose_output": "Choose Output Folder…",
        "clear_output": "Clear (use default)",
        "options_section": "4) Options",
        "restore": "Restore (for poor quality)",
        "modern": "Modern photo (current look)",
        "upscale": "Increase resolution",
        "upscale_amount": "Scale",
        "color_mode": "Color mode",
        "scale_2x": "2× larger",
        "scale_4x": "4× larger",
        "color_auto": "Automatic (recommended)",
        "color_bw": "Black & white",
        "color_keep": "Keep color",
        "run_start": "Start",
        "run_stop": "Stop",
        "status_idle": "Idle",
        "status_running": "Running…",
        "log_section": "Log",
        "lang_label": "Language:",
        "theme_label": "Appearance:",
        "theme_dark": "Dark",
        "theme_light": "Light",
        "theme_system": "System",
        "tip_restore": "Recommended for very old, blurry, or damaged photos.",
        "tip_modern": "Adjusts sharpness and contrast to look more modern.",
        "tip_upscale": "Enlarges the image without quality loss.",
        "tip_upscale_factor": "Improves resolution (2×: 1024 → 2048, 4×: 1024 → 4096).",
        "tip_choose_image": "Select a single image file.",
        "tip_choose_folder": "Select a folder to process all images inside.",
        "tip_choose_output": "Choose where outputs are saved.",
        "tip_clear_output": "Use default output folder next to input.",
        "missing_input_title": "Missing input",
        "missing_input_msg": "Please choose an image or folder.",
        "missing_key_title": "Missing API key",
        "missing_key_msg": "Please paste your Gemini API key, or create a .env file next to the app:\n\nGOOGLE_API_KEY=YOUR_KEY",
        "nothing_selected_title": "Nothing selected",
        "nothing_selected_msg": "Please select Restore and/or Modern 2025 and/or Upscale.",
        "missing_vertex_title": "Missing Vertex AI settings",
        "missing_vertex_msg": (
            "Upscaling uses Vertex AI. Please set these in your environment or .env file:\n\n"
            "GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID\n"
            "GOOGLE_CLOUD_LOCATION=YOUR_REGION (e.g., us-central1)\n\n"
            "Vertex AI also requires Application Default Credentials."
        ),
    },
    "de": {
        "title": "Foto Restore + Upscale (Gemini)",
        "settings_label": "Einstellungen",
        "menu_language": "Sprache",
        "menu_theme": "Darstellung",
        "menu_lang_de": "Deutsch",
        "menu_lang_en": "Englisch",
        "input_section": "2) Eingabe wählen (ein Bild oder Ordner)",
        "choose_image": "Bild wählen…",
        "choose_folder": "Ordner wählen…",
        "output_section": "3) Ausgabeordner (optional)",
        "choose_output": "Ausgabeordner wählen…",
        "clear_output": "Zurücksetzen (Standard)",
        "options_section": "4) Optionen",
        "restore": "Foto verbessern (für schlechte Qualität)",
        "modern": "Modernes Foto (aktueller Look)",
        "upscale": "Auflösung erhöhen",
        "upscale_amount": "Vergrößerung",
        "color_mode": "Farbmodus",
        "scale_2x": "2× größer",
        "scale_4x": "4× größer",
        "color_auto": "Automatisch (empfohlen)",
        "color_bw": "Schwarz-weiß",
        "color_keep": "Farbe beibehalten",
        "run_start": "Start",
        "run_stop": "Stopp",
        "status_idle": "Bereit",
        "status_running": "Läuft…",
        "log_section": "Log",
        "lang_label": "Sprache:",
        "theme_label": "Darstellung:",
        "theme_dark": "Dunkel",
        "theme_light": "Hell",
        "theme_system": "System",
        "tip_restore": "Empfohlen für sehr alte, unscharfe oder beschädigte Fotos.",
        "tip_modern": "Konvertiert das Foto in ein modernes Bild damit es aussieht wie heutzutage.",
        "tip_upscale": "Vergrößert das Bild ohne Qualitätsverlust.",
        "tip_upscale_factor": "Verbessert die Auflösung (2×: 1024 → 2048, 4×: 1024 → 4096).",
        "tip_choose_image": "Ein einzelnes Bild auswählen.",
        "tip_choose_folder": "Ordner wählen, um alle Bilder darin zu verarbeiten.",
        "tip_choose_output": "Zielordner für Ausgaben wählen.",
        "tip_clear_output": "Standard‑Ausgabeordner neben dem Eingang verwenden.",
        "missing_input_title": "Eingabe fehlt",
        "missing_input_msg": "Bitte ein Bild oder einen Ordner wählen.",
        "missing_key_title": "API‑Schlüssel fehlt",
        "missing_key_msg": "Bitte Gemini API‑Schlüssel einfügen oder eine .env Datei neben der App anlegen:\n\nGOOGLE_API_KEY=DEIN_KEY",
        "nothing_selected_title": "Nichts ausgewählt",
        "nothing_selected_msg": "Bitte Restore und/oder Modern 2025 und/oder Upscale auswählen.",
        "missing_vertex_title": "Vertex AI Einstellungen fehlen",
        "missing_vertex_msg": (
            "Upscaling nutzt Vertex AI. Bitte folgende Variablen in der Umgebung oder .env setzen:\n\n"
            "GOOGLE_CLOUD_PROJECT=DEIN_PROJEKT_ID\n"
            "GOOGLE_CLOUD_LOCATION=DEINE_REGION (z. B. europe-west3)\n\n"
            "Vertex AI benötigt außerdem Application Default Credentials."
        ),
    },
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.lang = tk.StringVar(value="de")
        self.theme = tk.StringVar(value="Light")
        self.title(UI_TEXT[self.lang.get()]["title"])
        self.geometry("940x680")
        self.minsize(940, 680)

        self.input_path = tk.StringVar(value="")
        self.output_path = tk.StringVar(value="")
        self.api_key = tk.StringVar(value=os.environ.get("GOOGLE_API_KEY", ""))

        self.do_restore = tk.BooleanVar(value=True)
        self.do_modern_2025 = tk.BooleanVar(value=False)
        self.do_upscale = tk.BooleanVar(value=False)

        self.scale_factor = tk.StringVar(value="2× größer")
        self.color_mode = tk.StringVar(value="Automatisch (empfohlen)")

        self.is_running = False
        self.worker_thread = None
        self.total_images = 0
        self._tooltips: dict[str, Tooltip] = {}
        self._settings_menu = None

        self._apply_theme()
        self._build_ui()
        self._build_menus()

    def _apply_theme(self):
        ctk.set_appearance_mode(self.theme.get())
        ctk.set_default_color_theme("blue")

    def t(self, key: str) -> str:
        return UI_TEXT[self.lang.get()].get(key, key)

    def _build_ui(self):
        pad = {"padx": 14, "pady": 10}

        # Top bar: app title + settings (taskbar style)
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=14, pady=(6, 2))

        self.lbl_settings = ctk.CTkLabel(
            top_bar, text=self.t("settings_label"), cursor="hand2"
        )
        self.lbl_settings.pack(side="left", padx=(6, 16), pady=6)
        self.lbl_settings.bind("<Button-1>", self._show_settings_menu)

        sep = ctk.CTkFrame(self, height=1)
        sep.pack(fill="x", padx=0, pady=(0, 8))

        # Input selection
        frm_in = ctk.CTkFrame(self)
        frm_in.pack(fill="x", **pad)

        self.lbl_in_title = ctk.CTkLabel(
            frm_in, text=self.t("input_section"), font=("Segoe UI Semibold", 12)
        )
        self.lbl_in_title.pack(anchor="w", padx=14, pady=(10, 6))

        row_in = ctk.CTkFrame(frm_in)
        row_in.pack(fill="x", padx=14, pady=(0, 12))

        self.entry_in = ctk.CTkEntry(row_in, textvariable=self.input_path)
        self.entry_in.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_choose_image = ctk.CTkButton(
            row_in, text=self.t("choose_image"), command=self.choose_image, width=140
        )
        self.btn_choose_image.pack(side="left", padx=(0, 8))
        self.btn_choose_folder = ctk.CTkButton(
            row_in, text=self.t("choose_folder"), command=self.choose_folder, width=140
        )
        self.btn_choose_folder.pack(side="left")
        self._attach_tooltip(self.btn_choose_image, "tip_choose_image")
        self._attach_tooltip(self.btn_choose_folder, "tip_choose_folder")

        # Output selection
        frm_out = ctk.CTkFrame(self)
        frm_out.pack(fill="x", **pad)

        self.lbl_out_title = ctk.CTkLabel(
            frm_out, text=self.t("output_section"), font=("Segoe UI Semibold", 12)
        )
        self.lbl_out_title.pack(anchor="w", padx=14, pady=(10, 6))

        row_out = ctk.CTkFrame(frm_out)
        row_out.pack(fill="x", padx=14, pady=(0, 12))

        self.entry_out = ctk.CTkEntry(row_out, textvariable=self.output_path)
        self.entry_out.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_choose_output = ctk.CTkButton(
            row_out, text=self.t("choose_output"), command=self.choose_output, width=170
        )
        self.btn_choose_output.pack(side="left", padx=(0, 8))
        self.btn_clear_output = ctk.CTkButton(
            row_out,
            text=self.t("clear_output"),
            command=lambda: self.output_path.set(""),
            width=160,
            fg_color="#1f2937",
            hover_color="#111827",
        )
        self.btn_clear_output.pack(side="left")
        self._attach_tooltip(self.btn_choose_output, "tip_choose_output")
        self._attach_tooltip(self.btn_clear_output, "tip_clear_output")

        # Options
        frm_opt = ctk.CTkFrame(self)
        frm_opt.pack(fill="x", **pad)

        self.lbl_opt_title = ctk.CTkLabel(
            frm_opt, text=self.t("options_section"), font=("Segoe UI Semibold", 12)
        )
        self.lbl_opt_title.pack(anchor="w", padx=14, pady=(10, 6))

        row1 = ctk.CTkFrame(frm_opt)
        row1.pack(fill="x", padx=14, pady=(0, 6))

        self.chk_restore = ctk.CTkCheckBox(
            row1, text=self.t("restore"), variable=self.do_restore
        )
        self.chk_restore.pack(side="left", padx=(0, 18))

        self.chk_modern = ctk.CTkCheckBox(
            row1, text=self.t("modern"), variable=self.do_modern_2025
        )
        self.chk_modern.pack(side="left", padx=(0, 18))

        self.chk_upscale = ctk.CTkCheckBox(
            row1, text=self.t("upscale"), variable=self.do_upscale
        )
        self.chk_upscale.pack(side="left")
        self._attach_tooltip(self.chk_restore, "tip_restore")
        self._attach_tooltip(self.chk_modern, "tip_modern")
        self._attach_tooltip(self.chk_upscale, "tip_upscale")

        row2 = ctk.CTkFrame(frm_opt)
        row2.pack(fill="x", padx=14, pady=(0, 12))

        self.lbl_scale = ctk.CTkLabel(row2, text=self.t("upscale_amount"))
        self.lbl_scale.pack(side="left")
        self.cbo_scale = ctk.CTkOptionMenu(
            row2, values=[self.t("scale_2x"), self.t("scale_4x")], variable=self.scale_factor, width=120
        )
        self.cbo_scale.pack(side="left", padx=8)
        self._attach_tooltip(self.lbl_scale, "tip_upscale_factor")
        self._attach_tooltip(self.cbo_scale, "tip_upscale_factor")

        self.lbl_color = ctk.CTkLabel(row2, text=self.t("color_mode"))
        self.lbl_color.pack(side="left", padx=(20, 0))
        self.cbo_color = ctk.CTkOptionMenu(
            row2,
            values=[self.t("color_auto"), self.t("color_bw"), self.t("color_keep")],
            variable=self.color_mode,
            width=200,
        )
        self.cbo_color.pack(side="left", padx=8)

        # Run controls
        frm_run = ctk.CTkFrame(self)
        frm_run.pack(fill="x", **pad)

        self.btn_start = ctk.CTkButton(
            frm_run, text=self.t("run_start"), command=self.start, width=120
        )
        self.btn_start.pack(side="left", padx=(14, 8))

        self.btn_stop = ctk.CTkButton(
            frm_run,
            text=self.t("run_stop"),
            command=self.stop,
            state="disabled",
            width=120,
            fg_color="#ef4444",
            hover_color="#dc2626",
        )
        self.btn_stop.pack(side="left")

        self.progress = ctk.CTkProgressBar(frm_run)
        self.progress.pack(side="left", padx=12, fill="x", expand=True)
        self.progress.set(0)

        self.lbl_status = ctk.CTkLabel(frm_run, text=self.t("status_idle"))
        self.lbl_status.pack(side="right", padx=14)

        # Log
        frm_log = ctk.CTkFrame(self)
        frm_log.pack(fill="both", expand=True, **pad)

        self.lbl_log_title = ctk.CTkLabel(
            frm_log, text=self.t("log_section"), font=("Segoe UI Semibold", 12)
        )
        self.lbl_log_title.pack(anchor="w", padx=14, pady=(10, 6))

        self.txt_log = tk.Text(
            frm_log,
            height=16,
            wrap="word",
            bg="#0b1220",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.txt_log.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self._apply_log_theme()

    def _set_language(self, code: str):
        self.lang.set(code)
        self._apply_texts()
        self._build_menus()

    def _set_theme(self, theme: str):
        self.theme.set(theme)
        ctk.set_appearance_mode(self.theme.get())
        self._apply_log_theme()

    def _build_menus(self):
        self._settings_menu = tk.Menu(self, tearoff=0, font=("Segoe UI", 11))
        lang_menu = tk.Menu(self._settings_menu, tearoff=0, font=("Segoe UI", 11))
        lang_menu.add_command(
            label=self.t("menu_lang_de"), command=lambda: self._set_language("de")
        )
        lang_menu.add_command(
            label=self.t("menu_lang_en"), command=lambda: self._set_language("en")
        )
        theme_menu = tk.Menu(self._settings_menu, tearoff=0, font=("Segoe UI", 11))
        theme_menu.add_command(
            label=self.t("theme_light"), command=lambda: self._set_theme("Light")
        )
        theme_menu.add_command(
            label=self.t("theme_dark"), command=lambda: self._set_theme("Dark")
        )
        theme_menu.add_command(
            label=self.t("theme_system"), command=lambda: self._set_theme("System")
        )
        self._settings_menu.add_cascade(label=self.t("menu_language"), menu=lang_menu)
        self._settings_menu.add_cascade(label=self.t("menu_theme"), menu=theme_menu)

    def _show_settings_menu(self, event):
        if self._settings_menu is None:
            self._build_menus()
        x = event.widget.winfo_rootx()
        y = event.widget.winfo_rooty() + event.widget.winfo_height() + 2
        self._settings_menu.tk_popup(x, y)

    def _apply_log_theme(self):
        theme = self.theme.get()
        if theme == "Light":
            bg = "#f8fafc"
            fg = "#0f172a"
        else:
            bg = "#0b1220"
            fg = "#e5e7eb"
        self.txt_log.configure(bg=bg, fg=fg, insertbackground=fg)

    def _apply_texts(self):
        self.title(self.t("title"))
        self.lbl_settings.configure(text=self.t("settings_label"))
        self.lbl_in_title.configure(text=self.t("input_section"))
        self.btn_choose_image.configure(text=self.t("choose_image"))
        self.btn_choose_folder.configure(text=self.t("choose_folder"))
        self.lbl_out_title.configure(text=self.t("output_section"))
        self.btn_choose_output.configure(text=self.t("choose_output"))
        self.btn_clear_output.configure(text=self.t("clear_output"))
        self.lbl_opt_title.configure(text=self.t("options_section"))
        self.chk_restore.configure(text=self.t("restore"))
        self.chk_modern.configure(text=self.t("modern"))
        self.chk_upscale.configure(text=self.t("upscale"))
        self.lbl_scale.configure(text=self.t("upscale_amount"))
        self.lbl_color.configure(text=self.t("color_mode"))
        self.cbo_scale.configure(values=[self.t("scale_2x"), self.t("scale_4x")])
        self.cbo_color.configure(
            values=[self.t("color_auto"), self.t("color_bw"), self.t("color_keep")]
        )
        if self.scale_factor.get() in {"2x", "2× größer", "2× larger"}:
            self.scale_factor.set(self.t("scale_2x"))
        elif self.scale_factor.get() in {"4x", "4× größer", "4× larger"}:
            self.scale_factor.set(self.t("scale_4x"))
        if self.color_mode.get() in {"Mixed", "Automatisch (empfohlen)"}:
            self.color_mode.set(self.t("color_auto"))
        elif self.color_mode.get() in {"Black & white", "Schwarz-weiß"}:
            self.color_mode.set(self.t("color_bw"))
        elif self.color_mode.get() in {"Color", "Farbe beibehalten"}:
            self.color_mode.set(self.t("color_keep"))
        self.btn_start.configure(text=self.t("run_start"))
        self.btn_stop.configure(text=self.t("run_stop"))
        self.lbl_status.configure(
            text=self.t("status_idle") if not self.is_running else self.t("status_running")
        )
        self.lbl_log_title.configure(text=self.t("log_section"))
        self._refresh_tooltips()

    def _attach_tooltip(self, widget, key: str):
        self._tooltips[key] = Tooltip(widget, self.t(key))

    def _refresh_tooltips(self):
        for key, tip in self._tooltips.items():
            tip.update_text(self.t(key))

    def log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.update_idletasks()

    def choose_image(self):
        filetypes = [("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff")]
        path = filedialog.askopenfilename(title="Select an image", filetypes=filetypes)
        if path:
            self.input_path.set(path)

    def choose_folder(self):
        path = filedialog.askdirectory(title="Select a folder with images")
        if path:
            self.input_path.set(path)

    def choose_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_path.set(path)

    def start(self):
        if self.is_running:
            return

        in_path = self.input_path.get().strip()
        if not in_path:
            messagebox.showerror(self.t("missing_input_title"), self.t("missing_input_msg"))
            return

        # Load .env first (so user can just double click exe and it works)
        load_dotenv()
        if not self.api_key.get().strip():
            self.api_key.set(os.environ.get("GOOGLE_API_KEY", ""))

        if not (self.do_restore.get() or self.do_modern_2025.get() or self.do_upscale.get()):
            messagebox.showerror(self.t("nothing_selected_title"), self.t("nothing_selected_msg"))
            return

        if self.do_restore.get() or self.do_modern_2025.get():
            if not self.api_key.get().strip():
                messagebox.showerror(self.t("missing_key_title"), self.t("missing_key_msg"))
                return

        if self.do_upscale.get():
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
            if not project or not location:
                messagebox.showerror(self.t("missing_vertex_title"), self.t("missing_vertex_msg"))
                return

        self.is_running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lbl_status.configure(text=self.t("status_running"))
        self.progress.set(0)

        self.worker_thread = threading.Thread(target=self._run_batch, daemon=True)
        self.worker_thread.start()

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.log("Stop requested. Finishing current image…")

    def _run_batch(self):
        try:
            # Make key available for the SDK
            if self.api_key.get().strip():
                os.environ["GOOGLE_API_KEY"] = self.api_key.get().strip()
                gemini_client = genai.Client()
            else:
                gemini_client = None

            vertex_client = None
            if self.do_upscale.get():
                project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
                location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
                vertex_client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=location,
                )

            input_path = Path(self.input_path.get().strip())
            if not input_path.exists():
                raise FileNotFoundError(f"Input not found: {input_path}")

            images = list_images(input_path)
            if not images:
                raise RuntimeError("No supported images found.")

            out_dir = ensure_output_folder(input_path, self.output_path.get().strip() or None)

            self.total_images = len(images)
            self.progress.set(0)
            self.log(f"Found {len(images)} image(s). Output folder: {out_dir}")

            scale = parse_scale_factor(self.scale_factor.get())
            mode = self.color_mode.get()

            for idx, img_path in enumerate(images, start=1):
                if not self.is_running:
                    break

                self.lbl_status.configure(text=f"{idx}/{len(images)}")

                try:
                    img = Image.open(img_path).convert("RGB")

                    # Build filename tag
                    suffix = []
                    if self.do_restore.get():
                        suffix.append("restored")
                    if self.do_modern_2025.get():
                        suffix.append("2025")
                    if self.do_upscale.get():
                        suffix.append(f"up{scale}x")
                    tag = "_".join(suffix) if suffix else "output"

                    out_path = out_dir / f"{img_path.stem}_{tag}.png"
                    if out_path.exists():
                        self.log(f"[{idx}] Skip (already exists): {out_path.name}")
                        if self.total_images:
                            self.progress.set(idx / self.total_images)
                        continue

                    # 1) Restore (recreate)
                    if self.do_restore.get():
                        self.log(f"[{idx}] Restore: {img_path.name}")
                        img = gemini_edit_image(
                            gemini_client, PROMPT_RESTORE_RECREATE, img, log=self.log
                        )
                        safe_sleep(0.8, self)

                    if not self.is_running:
                        break

                    # 2) Modern 2025 look (optional)
                    if self.do_modern_2025.get():
                        self.log(f"[{idx}] Modernize (2025 look): {img_path.name}")
                        img = gemini_edit_image(
                            gemini_client, PROMPT_MODERN_2025, img, log=self.log
                        )
                        safe_sleep(0.8, self)

                    if not self.is_running:
                        break

                    # 3) Upscale (second pass)
                    if self.do_upscale.get():
                        self.log(f"[{idx}] Upscale ({scale}x): {img_path.name}")
                        # Use Vertex AI upscaler (latest Imagen upscaling model)
                        upscale_factor = "x2" if scale == 2 else "x4"
                        img = vertex_upscale_image(vertex_client, img, upscale_factor)
                        safe_sleep(0.8, self)

                    # Save
                    img.save(out_path, format="PNG", optimize=True)
                    self.log(f"     Saved -> {out_path.name}")

                except Exception as e:
                    self.log(f"     ERROR on {img_path.name}: {e}")

                if self.total_images:
                    self.progress.set(idx / self.total_images)
                self.update_idletasks()

            if self.is_running:
                self.log("Done.")
            else:
                self.log("Stopped.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log(f"FATAL: {e}")
        finally:
            self.is_running = False
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.lbl_status.configure(text=self.t("status_idle"))


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
