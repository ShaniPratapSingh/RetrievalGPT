import os
import re
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from docx import Document
from src.core.observability import Logger

logger = Logger("multimodal")

# Try to import EasyOCR or Pytesseract for OCR support
HAS_OCR = False
OCR_READER = None

try:
    import easyocr
    OCR_READER = easyocr.Reader(['en'], gpu=False) # run CPU mode to ensure compatibility
    HAS_OCR = True
    logger.info("EasyOCR loaded successfully for multimodal image processing")
except Exception:
    try:
        import pytesseract
        HAS_OCR = True
        logger.info("Pytesseract loaded successfully for multimodal image processing")
    except Exception:
        logger.warn("No OCR engines (easyocr or pytesseract) could be loaded. Image inputs will run in simulated metadata mode.")


class MultiDocumentParser:
    @staticmethod
    def parse_pdf(file_path: str) -> List[Tuple[str, int]]:
        """Extract pages from a PDF document. Returns List of (text, page_number)."""
        pages = []
        try:
            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append((text, idx + 1))
        except Exception as e:
            logger.error("Failed to parse PDF", path=file_path, error=str(e))
        return pages

    @staticmethod
    def parse_docx(file_path: str) -> List[Tuple[str, int]]:
        """Extract paragraphs from a DOCX file. Returns List of (text, page_number)."""
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            
            combined = "\n".join(full_text)
            # Since docx does not store page counts directly, we chunk by roughly 1000 characters per "page"
            pages = []
            chunk_size = 1500
            for idx, i in enumerate(range(0, len(combined), chunk_size)):
                text_slice = combined[i:i+chunk_size]
                pages.append((text_slice, idx + 1))
            return pages
        except Exception as e:
            logger.error("Failed to parse DOCX", path=file_path, error=str(e))
            return []

    @staticmethod
    def parse_text_or_markdown(file_path: str) -> List[Tuple[str, int]]:
        """Extract text from TXT, MD, HTML files."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # If HTML, strip simple HTML tags
            if file_path.lower().endswith(".html"):
                content = re.sub(r'<[^>]*>', '', content)
                
            pages = []
            chunk_size = 1500
            for idx, i in enumerate(range(0, len(content), chunk_size)):
                text_slice = content[i:i+chunk_size]
                pages.append((text_slice, idx + 1))
            return pages
        except Exception as e:
            logger.error("Failed to parse text/markdown file", path=file_path, error=str(e))
            return []

    @staticmethod
    def parse_image_ocr(file_path: str) -> str:
        """Run OCR on target image file to extract embedded text or charts."""
        if not HAS_OCR:
            return (
                f"[IMAGE ANALYZED: {os.path.basename(file_path)}]\n"
                "OCR libraries are not configured. In a production environment, this would run EasyOCR/Tesseract. "
                "Simulated metadata: A chart displaying numerical information and performance metrics."
            )
            
        try:
            # 1. Try EasyOCR
            if OCR_READER:
                results = OCR_READER.readtext(file_path, detail=0)
                return "\n".join(results)
                
            # 2. Try Pytesseract
            import pytesseract
            from PIL import Image
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)
            
        except Exception as e:
            logger.error("OCR execution failed", path=file_path, error=str(e))
            return f"[OCR Failure on {os.path.basename(file_path)}]: {str(e)}"

    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Generate MD5 hash of the file for duplicate checking."""
        import hashlib
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
            return hasher.hexdigest()
        except Exception as e:
            logger.error("Hash calculation failed", path=file_path, error=str(e))
            return ""

    @staticmethod
    def extract_metadata(file_path: str, text: str) -> Dict[str, Any]:
        """Automatically extracts attributes from text content."""
        word_count = len(re.findall(r'\w+', text))
        reading_time = max(1, round(word_count / 200)) # Estimate 200 WPM
        
        # Detect basic language
        language = "en"
        if re.search(r'\b(der|die|das|und|ist)\b', text.lower()):
            language = "de"
        elif re.search(r'\b(el|la|los|las|y|es)\b', text.lower()):
            language = "es"
            
        return {
            "title": os.path.basename(file_path),
            "word_count": word_count,
            "reading_time_minutes": reading_time,
            "language": language
        }
