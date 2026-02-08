# Image Restoration & Upscale

Small desktop app to restore old photos and upscale images using Google Gemini and Vertex AI.

**Features**
- Restore damaged or low-quality photos.
- Modernize the look (optional).
- Upscale 2× or 4× using Vertex AI.
- Simple GUI with German/English and light/dark mode.

**Setup**
1. Create `.env` (copy from `.env.example`) and fill in your values.
2. Place your service account JSON file next to the app (or point to it in `.env`).
3. Install dependencies:
```bash
pip install -r requirements.txt
```
If you don’t have a `requirements.txt`, install:
```bash
pip install customtkinter google-genai pillow python-dotenv
```

**Run**
```bash
python photo_upscale_gui.py
```

**Notes**
- Gemini is used for Restore/Modern. Vertex AI is used for Upscale.
- Don’t commit real keys or service account files to GitHub.
