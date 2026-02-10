"""
Communication Tools - Translation and OCR (external APIs/libraries required).

Tools:
  - translate_text: Translate between languages (uses MyMemory API)
  - detect_language: Detect text language
  - ocr_image: Extract text from images (requires tesseract/easyocr)

Removed (AI can generate these):
  - email_draft → AI can write emails directly
  - summarize_text → AI can summarize natively
"""

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .tool_registry import Tool, RichParameter


class TranslateTextTool(Tool):
    """Translate text between languages using free API."""
    
    name = "translate_text"
    description = "Translate text from one language to another."
    parameters = {
        "text": "Text to translate",
        "target_language": "Target: en, es, fr, de, it, pt, ru, ja, ko, zh, ar",
        "source_language": "Source language (default: auto-detect)",
    }
    category = "communication"
    rich_parameters = [
        RichParameter(
            name="text",
            type="string",
            description="Text to translate (max 500 chars)",
            required=True,
        ),
        RichParameter(
            name="target_language",
            type="string",
            description="Target language code",
            required=True,
            enum=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"],
        ),
        RichParameter(
            name="source_language",
            type="string",
            description="Source language code (auto-detect if not specified)",
            required=False,
            default="auto",
        ),
    ]
    examples = [
        "translate_text(text='Hello world', target_language='es')",
        "translate_text(text='Bonjour', target_language='en', source_language='fr')",
    ]
    
    LANGUAGES = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
        'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
    }
    
    def execute(self, text: str, target_language: str, source_language: str = "auto", **kwargs) -> dict[str, Any]:
        try:
            # Use MyMemory API (free, no key)
            langpair = f"{source_language}|{target_language}"
            url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text[:500])}&langpair={langpair}"
            
            with urllib.request.urlopen(url, timeout=10) as r:
                data = json.loads(r.read().decode())
            
            if data.get("responseStatus") == 200:
                return {
                    "success": True,
                    "original": text,
                    "translated": data["responseData"]["translatedText"],
                    "source": source_language,
                    "target": target_language,
                    "target_name": self.LANGUAGES.get(target_language, target_language),
                }
            return {"success": False, "error": "Translation failed"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class DetectLanguageTool(Tool):
    """Detect text language."""
    
    name = "detect_language"
    description = "Detect what language text is written in."
    parameters = {"text": "Text to analyze"}
    category = "communication"
    rich_parameters = [
        RichParameter(
            name="text",
            type="string",
            description="Text to analyze for language detection",
            required=True,
        ),
    ]
    examples = [
        "detect_language(text='Hello how are you')",
        "detect_language(text='Bonjour comment allez-vous')",
    ]
    
    def execute(self, text: str, **kwargs) -> dict[str, Any]:
        try:
            # Try langdetect library
            try:
                from langdetect import detect, detect_langs
                lang = detect(text)
                probs = detect_langs(text)
                return {
                    "success": True,
                    "language": lang,
                    "confidence": [{"lang": str(p).split(':')[0], "prob": float(str(p).split(':')[1])} for p in probs[:3]],
                }
            except ImportError:
                pass
            
            # Simple heuristic fallback
            sample = text[:500]
            if re.search(r'[\u4e00-\u9fff]', sample): return {"success": True, "language": "zh", "confidence": [{"lang": "zh", "prob": 0.9}]}
            if re.search(r'[\u3040-\u30ff]', sample): return {"success": True, "language": "ja", "confidence": [{"lang": "ja", "prob": 0.9}]}
            if re.search(r'[\u0400-\u04ff]', sample): return {"success": True, "language": "ru", "confidence": [{"lang": "ru", "prob": 0.9}]}
            if re.search(r'[\u0600-\u06ff]', sample): return {"success": True, "language": "ar", "confidence": [{"lang": "ar", "prob": 0.9}]}
            if re.search(r'[\uac00-\ud7af]', sample): return {"success": True, "language": "ko", "confidence": [{"lang": "ko", "prob": 0.9}]}
            return {"success": True, "language": "en", "confidence": [{"lang": "en", "prob": 0.5}], "note": "Install langdetect for better accuracy"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class OCRImageTool(Tool):
    """Extract text from images using OCR."""
    
    name = "ocr_image"
    description = "Extract text from image using OCR. Requires tesseract or easyocr."
    parameters = {
        "path": "Path to image",
        "language": "OCR language (default: eng)",
    }
    category = "communication"
    rich_parameters = [
        RichParameter(
            name="path",
            type="string",
            description="Path to image file",
            required=True,
        ),
        RichParameter(
            name="language",
            type="string",
            description="OCR language code",
            required=False,
            default="eng",
            enum=["eng", "spa", "fra", "deu", "ita", "por", "rus", "jpn", "kor", "chi_sim"],
        ),
    ]
    examples = [
        "ocr_image(path='screenshot.png')",
        "ocr_image(path='document.jpg', language='deu')",
    ]
    
    def execute(self, path: str, language: str = "eng", **kwargs) -> dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            if not path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            # Try pytesseract
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(str(path))
                text = pytesseract.image_to_string(img, lang=language)
                return {"success": True, "text": text.strip(), "method": "tesseract", "words": len(text.split())}
            except ImportError:
                pass
            
            # Try easyocr
            try:
                import easyocr
                reader = easyocr.Reader([language.split('+')[0]])
                results = reader.readtext(str(path))
                text = "\n".join([r[1] for r in results])
                return {"success": True, "text": text, "method": "easyocr", "words": len(text.split())}
            except ImportError:
                pass
            
            # Try simple_ocr module
            try:
                from .simple_ocr import extract_text
                return {"success": True, "text": extract_text(str(path)), "method": "simple_ocr"}
            except ImportError:
                pass
            
            return {"success": False, "error": "No OCR library. Install: pip install pytesseract pillow"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Export
COMMUNICATION_TOOLS = [TranslateTextTool(), DetectLanguageTool(), OCRImageTool()]
def get_communication_tools(): return COMMUNICATION_TOOLS
