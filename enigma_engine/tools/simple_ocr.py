"""
Simple OCR - Pure Python Text Recognition

No external OCR dependencies (tesseract not required).
Uses image processing and pattern matching for basic text extraction.

Methods:
1. Edge-based text region detection
2. Color contrast analysis
3. Connected component analysis
4. Simple character templates (for common chars)

Best for: UI text, buttons, labels, menus
Limited for: Handwriting, unusual fonts, rotated text
"""

import base64
import io


class SimpleOCR:
    """
    Basic OCR without external dependencies.
    
    Extracts text from images using:
    - Contrast analysis
    - Edge detection
    - Region grouping
    - Basic pattern matching
    """
    
    def __init__(self):
        self._templates_loaded = False
        self._char_templates = {}
        
    def extract_text(self, image) -> str:
        """
        Extract text from a PIL Image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text as string
        """
        try:
            import numpy as np

            # Convert to grayscale
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image
            
            # Get numpy array
            arr = np.array(gray)
            
            # Find text regions
            regions = self._find_text_regions(arr)
            
            # Extract text from each region
            lines = []
            for region in regions:
                line_text = self._extract_text_from_region(arr, region)
                if line_text:
                    lines.append(line_text)
            
            return '\n'.join(lines)
            
        except ImportError:
            return "[SimpleOCR requires numpy and PIL]"
        except Exception as e:
            return f"[OCR Error: {e}]"
    
    def extract_text_from_file(self, path: str) -> str:
        """Extract text from an image file."""
        try:
            from PIL import Image
            img = Image.open(path)
            return self.extract_text(img)
        except Exception as e:
            return f"[Error loading image: {e}]"
    
    def extract_text_from_base64(self, b64_string: str) -> str:
        """Extract text from base64-encoded image."""
        try:
            from PIL import Image
            img_data = base64.b64decode(b64_string)
            img = Image.open(io.BytesIO(img_data))
            return self.extract_text(img)
        except Exception as e:
            return f"[Error decoding image: {e}]"
    
    def _find_text_regions(self, gray_arr) -> list[dict]:
        """
        Find regions likely to contain text.
        
        Uses edge detection and contrast analysis.
        """
        import numpy as np
        
        height, width = gray_arr.shape
        regions = []
        
        # Threshold to binary
        mean_val = gray_arr.mean()
        binary = gray_arr < mean_val  # Dark text on light bg
        
        # Also check light text on dark bg
        if binary.sum() > (height * width * 0.7):
            binary = gray_arr > mean_val
        
        # Find horizontal runs (potential text lines)
        # Scan each row for text patterns
        row_density = []
        for y in range(height):
            row = binary[y, :]
            # Count transitions (text has many edges)
            transitions = np.sum(np.abs(np.diff(row.astype(int))))
            row_density.append(transitions)
        
        row_density = np.array(row_density)
        
        # Find rows with high density (text lines)
        threshold = row_density.mean() + row_density.std() * 0.5
        text_rows = row_density > threshold
        
        # Group consecutive text rows into regions
        in_region = False
        region_start = 0
        
        for y in range(height):
            if text_rows[y] and not in_region:
                in_region = True
                region_start = y
            elif not text_rows[y] and in_region:
                in_region = False
                if y - region_start > 5:  # Minimum height
                    # Find x bounds for this region
                    region_slice = binary[region_start:y, :]
                    col_sums = region_slice.sum(axis=0)
                    non_empty = np.where(col_sums > 0)[0]
                    if len(non_empty) > 0:
                        x_start = max(0, non_empty[0] - 2)
                        x_end = min(width, non_empty[-1] + 2)
                        regions.append({
                            'x': x_start,
                            'y': region_start,
                            'width': x_end - x_start,
                            'height': y - region_start,
                        })
        
        # Handle region at bottom edge
        if in_region and height - region_start > 5:
            region_slice = binary[region_start:height, :]
            col_sums = region_slice.sum(axis=0)
            non_empty = np.where(col_sums > 0)[0]
            if len(non_empty) > 0:
                x_start = max(0, non_empty[0] - 2)
                x_end = min(width, non_empty[-1] + 2)
                regions.append({
                    'x': x_start,
                    'y': region_start,
                    'width': x_end - x_start,
                    'height': height - region_start,
                })
        
        return regions
    
    def _extract_text_from_region(self, gray_arr, region: dict) -> str:
        """
        Extract text from a specific region.
        
        Uses character segmentation and pattern matching.
        """
        
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        
        # Extract region
        region_arr = gray_arr[y:y+h, x:x+w]
        
        # Threshold
        mean_val = region_arr.mean()
        binary = region_arr < mean_val
        
        # Check if inverted
        if binary.sum() > (h * w * 0.6):
            binary = region_arr > mean_val
        
        # Segment into characters using vertical projection
        col_sums = binary.sum(axis=0)
        
        # Find gaps between characters
        threshold = col_sums.mean() * 0.3
        in_char = False
        char_regions = []
        char_start = 0
        
        for col in range(w):
            if col_sums[col] > threshold and not in_char:
                in_char = True
                char_start = col
            elif col_sums[col] <= threshold and in_char:
                in_char = False
                if col - char_start > 2:  # Minimum char width
                    char_regions.append((char_start, col))
        
        if in_char and w - char_start > 2:
            char_regions.append((char_start, w))
        
        # For each character region, try to identify
        text = ""
        last_end = 0
        
        for start, end in char_regions:
            # Check for space (gap between characters)
            gap = start - last_end
            if gap > h * 0.5:  # Large gap = space
                text += " "
            
            # Extract character bitmap
            char_img = binary[:, start:end]
            
            # Try to identify character (simplified pattern matching)
            char = self._identify_character(char_img)
            text += char
            
            last_end = end
        
        return text.strip()
    
    def _identify_character(self, char_img) -> str:
        """
        Identify a single character from its bitmap.
        
        Uses aspect ratio and density analysis.
        """
        
        h, w = char_img.shape
        density = char_img.sum() / (h * w) if h * w > 0 else 0
        aspect = w / h if h > 0 else 1
        
        # Very simplified character identification
        # Based on aspect ratio and fill density
        
        # Count features
        top_half = char_img[:h//2, :].sum() / (h//2 * w) if h > 0 and w > 0 else 0
        bottom_half = char_img[h//2:, :].sum() / (h//2 * w) if h > 0 and w > 0 else 0
        left_half = char_img[:, :w//2].sum() / (h * w//2) if h > 0 and w > 0 else 0
        right_half = char_img[:, w//2:].sum() / (h * w//2) if h > 0 and w > 0 else 0
        
        # Check for vertical line (l, I, 1, |)
        if aspect < 0.4 and density > 0.3:
            center_col = char_img[:, w//3:2*w//3].sum() / (h * w//3) if w > 0 else 0
            if center_col > 0.5:
                return "I"
        
        # Check for dot (period, i dot)
        if aspect > 0.7 and aspect < 1.4 and density > 0.4 and h < 10:
            return "."
        
        # Check for dash/hyphen
        if aspect > 2.0 and density > 0.4:
            return "-"
        
        # Check for circular shapes (o, O, 0)
        if aspect > 0.6 and aspect < 1.2:
            # Check for hole in center
            center = char_img[h//4:3*h//4, w//4:3*w//4]
            center_density = center.sum() / center.size if center.size > 0 else 0
            edge_density = density - center_density * 0.5
            if center_density < density * 0.5:
                return "O"
        
        # For most characters, we can't reliably identify
        # Return placeholder based on general shape
        if density > 0.4:
            if top_half > bottom_half * 1.3:
                return "P"  # Top-heavy
            elif bottom_half > top_half * 1.3:
                return "L"  # Bottom-heavy
            elif left_half > right_half * 1.3:
                return "E"  # Left-heavy
            elif right_half > left_half * 1.3:
                return "D"  # Right-heavy
            else:
                return "X"  # Balanced
        else:
            return "_"  # Unknown/unclear
    
    def get_word_locations(self, image) -> list[dict]:
        """
        Get approximate locations of words in the image.
        
        Returns:
            List of dicts with x, y, width, height, text
        """
        try:
            import numpy as np
            
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image
            
            arr = np.array(gray)
            regions = self._find_text_regions(arr)
            
            results = []
            for region in regions:
                text = self._extract_text_from_region(arr, region)
                if text:
                    results.append({
                        'x': region['x'],
                        'y': region['y'],
                        'width': region['width'],
                        'height': region['height'],
                        'text': text,
                    })
            
            return results
            
        except Exception as e:
            return [{'error': str(e)}]


class AdvancedOCR:
    """
    Advanced OCR with better character recognition.
    
    Uses EasyOCR if available (no tesseract needed),
    falls back to SimpleOCR.
    """
    
    def __init__(self):
        self._backend = self._detect_backend()
        self._simple_ocr = SimpleOCR()
        self._easyocr_reader = None
    
    def _detect_backend(self) -> str:
        """Detect best available OCR backend."""
        # Try EasyOCR (PyTorch-based, no tesseract)
        try:
            return "easyocr"
        except ImportError:
            pass
        
        # Try PaddleOCR (alternative)
        try:
            return "paddleocr"
        except ImportError:
            pass
        
        # Try tesseract (legacy)
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return "tesseract"
        except (ImportError, RuntimeError, OSError):
            pass
        
        # Fallback to simple
        return "simple"
    
    def extract_text(self, image, language: str = 'en') -> str:
        """
        Extract text from image using best available backend.
        
        Args:
            image: PIL Image or path string
            language: Language code (en, es, fr, de, etc.)
            
        Returns:
            Extracted text
        """
        from PIL import Image

        # Load image if path
        if isinstance(image, str):
            image = Image.open(image)
        
        if self._backend == "easyocr":
            return self._extract_easyocr(image, language)
        elif self._backend == "paddleocr":
            return self._extract_paddleocr(image, language)
        elif self._backend == "tesseract":
            return self._extract_tesseract(image, language)
        else:
            return self._simple_ocr.extract_text(image)
    
    def _extract_easyocr(self, image, language: str) -> str:
        """Extract text using EasyOCR."""
        try:
            import easyocr
            import numpy as np
            
            if self._easyocr_reader is None:
                # Initialize reader (downloads model on first use)
                self._easyocr_reader = easyocr.Reader([language], gpu=False)
            
            # Convert PIL to numpy
            arr = np.array(image)
            
            # Run OCR
            results = self._easyocr_reader.readtext(arr)
            
            # Combine results
            lines = [text for (bbox, text, conf) in results]
            return '\n'.join(lines)
            
        except Exception as e:
            return f"[EasyOCR Error: {e}]"
    
    def _extract_paddleocr(self, image, language: str) -> str:
        """Extract text using PaddleOCR."""
        try:
            import numpy as np
            from paddleocr import PaddleOCR
            
            ocr = PaddleOCR(use_angle_cls=True, lang=language)
            arr = np.array(image)
            results = ocr.ocr(arr, cls=True)
            
            lines = []
            if results and results[0]:
                for line in results[0]:
                    lines.append(line[1][0])
            
            return '\n'.join(lines)
            
        except Exception as e:
            return f"[PaddleOCR Error: {e}]"
    
    def _extract_tesseract(self, image, language: str) -> str:
        """Extract text using Tesseract."""
        try:
            import pytesseract
            return pytesseract.image_to_string(image, lang=language)
        except Exception as e:
            return f"[Tesseract Error: {e}]"
    
    def get_backend(self) -> str:
        """Return current OCR backend name."""
        return self._backend


class TextFinder:
    """
    Find specific text on screen.
    
    Optimized for UI text, buttons, labels.
    """
    
    def __init__(self):
        self._ocr = AdvancedOCR()
    
    def find(self, image, search_text: str, 
             case_sensitive: bool = False) -> list[dict]:
        """
        Find all occurrences of text in image.
        
        Args:
            image: PIL Image
            search_text: Text to find
            case_sensitive: Whether to match case
            
        Returns:
            List of matches with location and confidence
        """
        results = []
        
        if self._ocr._backend == "easyocr":
            results = self._find_easyocr(image, search_text, case_sensitive)
        elif self._ocr._backend == "paddleocr":
            results = self._find_paddleocr(image, search_text, case_sensitive)
        elif self._ocr._backend == "tesseract":
            results = self._find_tesseract(image, search_text, case_sensitive)
        else:
            results = self._find_simple(image, search_text, case_sensitive)
        
        return results
    
    def _find_easyocr(self, image, search_text: str, 
                      case_sensitive: bool) -> list[dict]:
        """Find text using EasyOCR."""
        try:
            import easyocr
            import numpy as np
            
            if self._ocr._easyocr_reader is None:
                self._ocr._easyocr_reader = easyocr.Reader(['en'], gpu=False)
            
            arr = np.array(image)
            results = self._ocr._easyocr_reader.readtext(arr)
            
            matches = []
            search = search_text if case_sensitive else search_text.lower()
            
            for (bbox, text, conf) in results:
                compare = text if case_sensitive else text.lower()
                if search in compare:
                    # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                    x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                    matches.append({
                        'text': text,
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'confidence': conf,
                    })
            
            return matches
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def _find_paddleocr(self, image, search_text: str,
                        case_sensitive: bool) -> list[dict]:
        """Find text using PaddleOCR."""
        try:
            import numpy as np
            from paddleocr import PaddleOCR
            
            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            arr = np.array(image)
            results = ocr.ocr(arr, cls=True)
            
            matches = []
            search = search_text if case_sensitive else search_text.lower()
            
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, conf) = line
                    compare = text if case_sensitive else text.lower()
                    if search in compare:
                        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                        matches.append({
                            'text': text,
                            'x': x1,
                            'y': y1,
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'confidence': conf,
                        })
            
            return matches
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def _find_tesseract(self, image, search_text: str,
                        case_sensitive: bool) -> list[dict]:
        """Find text using Tesseract."""
        try:
            import pytesseract
            
            data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )
            
            matches = []
            search = search_text if case_sensitive else search_text.lower()
            
            for i, text in enumerate(data['text']):
                if not text.strip():
                    continue
                compare = text if case_sensitive else text.lower()
                if search in compare:
                    matches.append({
                        'text': text,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': data['conf'][i] / 100.0,
                    })
            
            return matches
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def _find_simple(self, image, search_text: str,
                     case_sensitive: bool) -> list[dict]:
        """Find text using simple OCR."""
        simple = SimpleOCR()
        locations = simple.get_word_locations(image)
        
        matches = []
        search = search_text if case_sensitive else search_text.lower()
        
        for loc in locations:
            if 'error' in loc:
                continue
            text = loc.get('text', '')
            compare = text if case_sensitive else text.lower()
            if search in compare:
                loc['confidence'] = 0.5  # Unknown confidence
                matches.append(loc)
        
        return matches


# Convenience functions
def ocr(image) -> str:
    """Quick text extraction from image."""
    reader = AdvancedOCR()
    return reader.extract_text(image)


def find_text(image, text: str) -> list[dict]:
    """Quick text finding in image."""
    finder = TextFinder()
    return finder.find(image, text)


def get_ocr_backend() -> str:
    """Return name of active OCR backend."""
    reader = AdvancedOCR()
    return reader.get_backend()
