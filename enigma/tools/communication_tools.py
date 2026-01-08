"""
Communication Tools - Email drafting, translation, summarization, OCR.

Tools:
  - email_draft: Draft an email given context
  - translate_text: Translate between languages
  - summarize_text: Summarize long text
  - ocr_image: Extract text from images
  - detect_language: Detect the language of text
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from .tool_registry import Tool

# Email drafts storage
EMAIL_DRAFTS_DIR = Path.home() / ".enigma" / "email_drafts"
EMAIL_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# EMAIL TOOLS
# ============================================================================

class EmailDraftTool(Tool):
    """Draft an email."""
    
    name = "email_draft"
    description = "Draft an email based on context. Generates professional email text."
    parameters = {
        "to": "Recipient name or role (e.g., 'boss', 'client', 'John')",
        "subject": "Email subject",
        "context": "What the email should be about (e.g., 'request time off next week')",
        "tone": "Tone: 'formal', 'casual', 'friendly', 'professional' (default: professional)",
        "your_name": "Your name for the signature (optional)",
    }
    
    def execute(self, to: str, subject: str, context: str, 
                tone: str = "professional", your_name: str = None, **kwargs) -> Dict[str, Any]:
        try:
            # Build email template based on tone
            greetings = {
                "formal": f"Dear {to},",
                "professional": f"Hi {to},",
                "casual": f"Hey {to},",
                "friendly": f"Hi {to}!",
            }
            
            closings = {
                "formal": "Sincerely,",
                "professional": "Best regards,",
                "casual": "Thanks,",
                "friendly": "Cheers,",
            }
            
            greeting = greetings.get(tone, greetings["professional"])
            closing = closings.get(tone, closings["professional"])
            
            # Generate body (placeholder - in real use, an LLM would generate this)
            body = self._generate_body(context, tone)
            
            # Assemble email
            email_text = f"""Subject: {subject}

{greeting}

{body}

{closing}
{your_name or '[Your Name]'}"""
            
            # Save draft
            draft_file = EMAIL_DRAFTS_DIR / f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(draft_file, 'w') as f:
                f.write(email_text)
            
            return {
                "success": True,
                "email": email_text,
                "draft_saved": str(draft_file),
                "note": "Review and customize before sending",
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_body(self, context: str, tone: str) -> str:
        """Generate email body from context."""
        # Simple template-based generation
        # In practice, this would use the AI model
        
        templates = {
            "request time off": "I am writing to request time off {context}. I have ensured that my current tasks are up to date and have arranged for coverage during my absence.\n\nPlease let me know if you need any additional information.",
            "follow up": "I wanted to follow up on {context}. I'm keen to hear your thoughts and move forward with this.\n\nPlease let me know if you have any questions.",
            "introduction": "I hope this email finds you well. {context}\n\nI would be happy to discuss this further at your convenience.",
            "thank you": "Thank you for {context}. I really appreciate your time and effort.\n\nPlease don't hesitate to reach out if there's anything I can help with.",
            "meeting": "I would like to schedule a meeting to discuss {context}. Please let me know your availability, and I'll send a calendar invite.\n\nLooking forward to speaking with you.",
        }
        
        # Find matching template
        context_lower = context.lower()
        for key, template in templates.items():
            if key in context_lower:
                return template.format(context=context)
        
        # Default template
        return f"I am writing regarding {context}.\n\nPlease let me know if you have any questions or need additional information."


class ListEmailDraftsTool(Tool):
    """List saved email drafts."""
    
    name = "list_email_drafts"
    description = "List all saved email drafts."
    parameters = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            drafts = []
            for file in sorted(EMAIL_DRAFTS_DIR.glob("draft_*.txt"), reverse=True)[:20]:
                with open(file, 'r') as f:
                    content = f.read()
                # Extract subject
                subject_match = re.search(r'Subject: (.+)', content)
                subject = subject_match.group(1) if subject_match else "No subject"
                
                drafts.append({
                    "file": file.name,
                    "subject": subject,
                    "created": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                })
            
            return {
                "success": True,
                "count": len(drafts),
                "drafts": drafts,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# TRANSLATION TOOLS
# ============================================================================

class TranslateTextTool(Tool):
    """Translate text between languages."""
    
    name = "translate_text"
    description = "Translate text from one language to another using free translation APIs."
    parameters = {
        "text": "The text to translate",
        "target_language": "Target language code (e.g., 'en', 'es', 'fr', 'de', 'ja', 'zh')",
        "source_language": "Source language code (default: auto-detect)",
    }
    
    # Language code mapping
    LANGUAGES = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
        'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
        'nl': 'Dutch', 'pl': 'Polish', 'sv': 'Swedish', 'tr': 'Turkish',
    }
    
    def execute(self, text: str, target_language: str, 
                source_language: str = "auto", **kwargs) -> Dict[str, Any]:
        try:
            import urllib.request
            import urllib.parse
            
            # Use LibreTranslate (free, self-hostable)
            # Fall back to MyMemory API (free tier)
            
            translated = None
            method_used = ""
            
            # Try MyMemory first (no key needed)
            try:
                langpair = f"{source_language}|{target_language}"
                url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text)}&langpair={langpair}"
                
                with urllib.request.urlopen(url, timeout=10) as response:
                    data = json.loads(response.read().decode())
                
                if data.get("responseStatus") == 200:
                    translated = data["responseData"]["translatedText"]
                    method_used = "MyMemory"
                    
            except Exception as e:
                pass
            
            if not translated:
                return {
                    "success": False, 
                    "error": "Translation failed. Try installing: pip install deep-translator"
                }
            
            return {
                "success": True,
                "original": text,
                "translated": translated,
                "source_language": source_language,
                "target_language": target_language,
                "target_language_name": self.LANGUAGES.get(target_language, target_language),
                "method": method_used,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class DetectLanguageTool(Tool):
    """Detect the language of text."""
    
    name = "detect_language"
    description = "Detect what language a piece of text is written in."
    parameters = {
        "text": "The text to analyze",
    }
    
    def execute(self, text: str, **kwargs) -> Dict[str, Any]:
        try:
            # Try langdetect library
            try:
                from langdetect import detect, detect_langs
                
                language = detect(text)
                probabilities = detect_langs(text)
                
                return {
                    "success": True,
                    "detected_language": language,
                    "confidence": [{"lang": str(p).split(':')[0], "prob": float(str(p).split(':')[1])} 
                                  for p in probabilities[:3]],
                }
            except ImportError:
                pass
            
            # Simple heuristic fallback
            # Check for common character patterns
            text_sample = text[:500]
            
            if re.search(r'[\u4e00-\u9fff]', text_sample):
                return {"success": True, "detected_language": "zh", "confidence": [{"lang": "zh", "prob": 0.9}]}
            elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text_sample):
                return {"success": True, "detected_language": "ja", "confidence": [{"lang": "ja", "prob": 0.9}]}
            elif re.search(r'[\u0400-\u04ff]', text_sample):
                return {"success": True, "detected_language": "ru", "confidence": [{"lang": "ru", "prob": 0.9}]}
            elif re.search(r'[\u0600-\u06ff]', text_sample):
                return {"success": True, "detected_language": "ar", "confidence": [{"lang": "ar", "prob": 0.9}]}
            elif re.search(r'[\uac00-\ud7af]', text_sample):
                return {"success": True, "detected_language": "ko", "confidence": [{"lang": "ko", "prob": 0.9}]}
            else:
                # Assume English for Latin script
                return {"success": True, "detected_language": "en", "confidence": [{"lang": "en", "prob": 0.5}], 
                        "note": "For better accuracy, install: pip install langdetect"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# SUMMARIZATION TOOLS
# ============================================================================

class SummarizeTextTool(Tool):
    """Summarize long text."""
    
    name = "summarize_text"
    description = "Summarize a long piece of text into key points or a shorter version."
    parameters = {
        "text": "The text to summarize",
        "max_sentences": "Maximum sentences in summary (default: 5)",
        "style": "Style: 'bullets', 'paragraph', 'tldr' (default: bullets)",
    }
    
    def execute(self, text: str, max_sentences: int = 5, 
                style: str = "bullets", **kwargs) -> Dict[str, Any]:
        try:
            # Try transformers summarization first
            try:
                from transformers import pipeline
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                
                # Handle long texts
                max_length = 1024
                if len(text) > max_length * 4:
                    text = text[:max_length * 4]
                
                summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
                summary_text = summary[0]['summary_text']
                
                return {
                    "success": True,
                    "summary": summary_text,
                    "method": "BART",
                    "original_length": len(text),
                    "summary_length": len(summary_text),
                }
                
            except ImportError:
                pass
            
            # Fallback: Extractive summarization using sentence scoring
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            if len(sentences) <= max_sentences:
                return {
                    "success": True,
                    "summary": text,
                    "note": "Text was already short enough",
                }
            
            # Score sentences by position and word frequency
            word_freq = {}
            for sentence in sentences:
                words = re.findall(r'\b\w+\b', sentence.lower())
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score each sentence
            scored = []
            for i, sentence in enumerate(sentences):
                words = re.findall(r'\b\w+\b', sentence.lower())
                score = sum(word_freq.get(w, 0) for w in words) / max(len(words), 1)
                # Boost first sentences
                if i < 3:
                    score *= 1.5
                scored.append((score, i, sentence))
            
            # Get top sentences, maintaining order
            top_sentences = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
            top_sentences = sorted(top_sentences, key=lambda x: x[1])
            
            if style == "bullets":
                summary = "\n".join(f"• {s[2].strip()}" for s in top_sentences)
            elif style == "tldr":
                summary = "TL;DR: " + " ".join(s[2].strip() for s in top_sentences[:2])
            else:  # paragraph
                summary = " ".join(s[2].strip() for s in top_sentences)
            
            return {
                "success": True,
                "summary": summary,
                "method": "extractive",
                "original_length": len(text),
                "summary_length": len(summary),
                "original_sentences": len(sentences),
                "summary_sentences": len(top_sentences),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# OCR TOOLS
# ============================================================================

class OCRImageTool(Tool):
    """Extract text from images using OCR."""
    
    name = "ocr_image"
    description = "Extract text from an image file using OCR (Optical Character Recognition)."
    parameters = {
        "path": "Path to the image file",
        "language": "OCR language (default: eng). Use 'eng+fra' for multiple.",
        "preprocess": "Preprocessing: 'none', 'threshold', 'blur' (default: none)",
    }
    
    def execute(self, path: str, language: str = "eng", 
                preprocess: str = "none", **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            # Try pytesseract (most common)
            try:
                import pytesseract
                from PIL import Image
                import cv2
                import numpy as np
                
                # Load image
                img = cv2.imread(str(path))
                if img is None:
                    img = np.array(Image.open(str(path)))
                
                # Convert to grayscale
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                # Preprocessing
                if preprocess == "threshold":
                    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                elif preprocess == "blur":
                    gray = cv2.medianBlur(gray, 3)
                
                # OCR
                text = pytesseract.image_to_string(gray, lang=language)
                
                # Also get confidence data
                data = pytesseract.image_to_data(gray, lang=language, output_type=pytesseract.Output.DICT)
                confidences = [int(c) for c in data['conf'] if int(c) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                return {
                    "success": True,
                    "text": text.strip(),
                    "method": "tesseract",
                    "language": language,
                    "confidence": round(avg_confidence, 1),
                    "word_count": len(text.split()),
                }
                
            except ImportError:
                pass
            
            # Try easyocr
            try:
                import easyocr
                
                reader = easyocr.Reader([language.split('+')[0]])
                results = reader.readtext(str(path))
                
                text = "\n".join([r[1] for r in results])
                avg_confidence = sum(r[2] for r in results) / len(results) if results else 0
                
                return {
                    "success": True,
                    "text": text,
                    "method": "easyocr",
                    "confidence": round(avg_confidence * 100, 1),
                    "word_count": len(text.split()),
                }
                
            except ImportError:
                pass
            
            # Fallback: Use simple_ocr module if available
            try:
                from .simple_ocr import extract_text
                text = extract_text(str(path))
                return {
                    "success": True,
                    "text": text,
                    "method": "simple_ocr",
                }
            except ImportError:
                pass
            
            return {
                "success": False,
                "error": "No OCR library available. Install: pip install pytesseract pillow opencv-python"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class OCRScreenshotTool(Tool):
    """Take a screenshot and extract text from it."""
    
    name = "ocr_screenshot"
    description = "Take a screenshot and extract text from it using OCR."
    parameters = {
        "region": "Optional region as 'x,y,width,height'. Default: full screen",
        "language": "OCR language (default: eng)",
    }
    
    def execute(self, region: str = None, language: str = "eng", **kwargs) -> Dict[str, Any]:
        try:
            import tempfile
            
            # Take screenshot
            from .system_tools import ScreenshotTool
            screenshot_tool = ScreenshotTool()
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            
            result = screenshot_tool.execute(output_path=temp_path, region=region)
            
            if not result.get('success'):
                return result
            
            # OCR the screenshot
            ocr_tool = OCRImageTool()
            ocr_result = ocr_tool.execute(path=temp_path, language=language)
            
            # Cleanup
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return ocr_result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
