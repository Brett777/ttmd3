"""
Image conversion utilities for RAG-Ultra: convert document pages to base64-encoded JPEG images.
Supports PDFs (via PyMuPDF or pdf2image), PPTX (via conversion), and robust parallel processing.
"""

import os
import base64
import tempfile
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import io

from .utils import logger

# Optional dependency imports at module level
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Default to lower DPI for better performance
DEFAULT_DPI = 50
# Default number of workers for parallel processing
DEFAULT_MAX_WORKERS = 8
DEFAULT_JPEG_QUALITY = 40

def convert_page_to_image(document_path: str, page_num: int, dpi: int = DEFAULT_DPI, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> str:
    """
    Convert a specific page from a document to a base64-encoded JPEG image.
    
    Args:
        document_path: Path to the document
        page_num: Page number to convert (1-indexed)
        dpi: Resolution in dots per inch
        jpeg_quality: JPEG compression quality (1-95)
        
    Returns:
        Base64-encoded string of the JPEG image
    
    Raises:
        ValueError: If document type is not supported or the page cannot be converted
    """
    file_ext = os.path.splitext(document_path)[1].lower().lstrip('.')
    
    if file_ext == "pdf":
        # Prefer PyMuPDF (faster) and only fall back when necessary
        if FITZ_AVAILABLE:
            result = convert_pdf_page_to_image_fitz(document_path, page_num, dpi, jpeg_quality)
            if result:
                return result
                
        if PDF2IMAGE_AVAILABLE:
            return convert_pdf_page_to_image(document_path, page_num, dpi, jpeg_quality)
        
        logger.error("No PDF conversion library available. Install PyMuPDF or pdf2image.")
        return ""
    elif file_ext == "docx":
        logger.warning("Direct DOCX to image conversion is not supported. Consider converting to PDF first.")
        return ""
    elif file_ext == "pptx":
        return convert_pptx_slide_to_image(document_path, page_num, dpi, jpeg_quality)
    elif file_ext == "txt":
        logger.warning("Text files cannot be directly converted to images.")
        return ""
    else:
        raise ValueError(f"Unsupported file type for image conversion: {file_ext}")

def _process_pdf_page_fitz(doc_path: str, page_idx: int, zoom: float, jpeg_quality: int) -> tuple:
    """
    Helper for parallel PDF page conversion using PyMuPDF.
    Returns (1-indexed page number, base64 JPEG string).
    """
    try:
        if not FITZ_AVAILABLE or not PIL_AVAILABLE:
            return (page_idx, "")
        doc = fitz.open(doc_path)
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return (page_idx + 1, img_base64)
    except Exception as e:
        logger.error(f"Error processing page {page_idx+1}: {e}")
        return (page_idx + 1, "")

def convert_pdf_page_to_image_fitz(pdf_path: str, page_num: int, dpi: int = DEFAULT_DPI, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> str:
    """
    Convert a PDF page to a base64-encoded JPEG image using PyMuPDF (fitz).
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to convert (1-indexed)
        dpi: Resolution in dots per inch
        jpeg_quality: JPEG compression quality (1-95)
        
    Returns:
        Base64-encoded string of the JPEG image
    """
    if not FITZ_AVAILABLE or not PIL_AVAILABLE:
        logger.warning("PyMuPDF or Pillow not installed. Install with 'pip install PyMuPDF Pillow'")
        return ""
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Check if page number is valid
        if page_num < 1 or page_num > len(doc):
            logger.error(f"Invalid page number {page_num}. PDF has {len(doc)} pages.")
            return ""
        
        # Get the page (0-indexed in PyMuPDF)
        page = doc[page_num - 1]
        
        # Calculate zoom factor based on DPI
        zoom = dpi / 72  # 72 is the default DPI for PDF
        
        # Get the pixmap (image)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        logger.info(f"Converted page {page_num} of {pdf_path} to base64 JPEG image using PyMuPDF")
        return img_base64
    
    except Exception as e:
        logger.error(f"Error converting PDF page to image using PyMuPDF: {e}")
        return ""

def convert_pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = DEFAULT_DPI, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> str:
    """
    Convert a PDF page to a base64-encoded JPEG image using pdf2image.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to convert (1-indexed)
        dpi: Resolution in dots per inch
        jpeg_quality: JPEG compression quality (1-95)
        
    Returns:
        Base64-encoded string of the JPEG image
    """
    if not PDF2IMAGE_AVAILABLE or not PIL_AVAILABLE:
        logger.error("pdf2image and Pillow libraries are required. Install with 'pip install pdf2image Pillow'")
        return ""
    
    try:
        # Convert the PDF page to image
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num
        )
        
        if not images:
            logger.error(f"Failed to convert page {page_num} of PDF {pdf_path}")
            return ""
        
        # Get the first image (should be the only one)
        image = images[0]
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        logger.info(f"Converted page {page_num} of {pdf_path} to base64 JPEG image")
        return img_base64
    except Exception as e:
        logger.error(f"Error converting PDF page to image: {e}")
        return ""

def convert_pptx_slide_to_image(pptx_path: str, slide_num: int, dpi: int = DEFAULT_DPI, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> str:
    """
    Convert a PowerPoint slide to a base64-encoded JPEG image.
    
    This requires a combination of libraries and may use an external converter.
    
    Args:
        pptx_path: Path to the PowerPoint file
        slide_num: Slide number to convert (1-indexed)
        dpi: Resolution in dots per inch
        jpeg_quality: JPEG compression quality (1-95)
        
    Returns:
        Base64-encoded string of the JPEG image
    """
    if not PIL_AVAILABLE:
        logger.error("Pillow library is required. Install with 'pip install Pillow'")
        return ""
    
    try:
        import subprocess
        
        # Check if libreoffice or unoconv is available (need one for conversion)
        use_libreoffice = False
        use_unoconv = False
        
        try:
            subprocess.run(["libreoffice", "--version"], check=True, capture_output=True)
            use_libreoffice = True
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                subprocess.run(["unoconv", "--version"], check=True, capture_output=True)
                use_unoconv = True
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.error("Neither LibreOffice nor unoconv found for PPTX conversion")
                return ""
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_output = os.path.join(temp_dir, "output.pdf")
            
            # Convert PPTX to PDF
            if use_libreoffice:
                subprocess.run([
                    "libreoffice", "--headless", "--convert-to", "pdf",
                    "--outdir", temp_dir, pptx_path
                ], check=True, capture_output=True)
                
                # Rename if necessary
                orig_name = os.path.splitext(os.path.basename(pptx_path))[0] + ".pdf"
                orig_path = os.path.join(temp_dir, orig_name)
                if orig_path != pdf_output:
                    os.rename(orig_path, pdf_output)
            
            elif use_unoconv:
                subprocess.run([
                    "unoconv", "-f", "pdf", "-o", pdf_output, pptx_path
                ], check=True, capture_output=True)
            
            # Now convert the PDF page to image
            if os.path.exists(pdf_output):
                return convert_pdf_page_to_image(pdf_output, slide_num, dpi, jpeg_quality)
            else:
                logger.error(f"Failed to convert PPTX to PDF: {pptx_path}")
                return ""
    except Exception as e:
        logger.error(f"Error converting PPTX slide to image: {e}")
        return ""

def convert_document_pages_to_images(
    document_path: str,
    dpi: int = DEFAULT_DPI,
    poppler_path: str = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    callback: Optional[Callable[[int], None]] = None,
) -> Dict[int, str]:
    """
    Convert all pages in a document to base64-encoded JPEG images.
    
    Args:
        document_path: Path to the document
        dpi: Resolution in dots per inch
        poppler_path: Path to poppler binaries (required for Windows with pdf2image)
        max_workers: Maximum number of parallel processes to use
        jpeg_quality: JPEG compression quality (1-95)
        callback: Optional callback function to invoke after each page is processed
        
    Returns:
        Dict mapping page numbers to base64-encoded JPEG images
    """
    file_ext = os.path.splitext(document_path)[1].lower().lstrip('.')
    
    # For now, only fully support PDF for bulk conversion
    if file_ext == "pdf":
        # Prefer PyMuPDF (faster) if available
        if FITZ_AVAILABLE and PIL_AVAILABLE:
            try:
                return convert_pdf_to_images_fitz(document_path, dpi, max_workers, jpeg_quality, callback)
            except Exception as e:
                logger.warning(f"Failed to convert PDF with PyMuPDF: {e}. Trying pdf2image...")
        
        # Fallback to pdf2image
        if PDF2IMAGE_AVAILABLE:
            return convert_pdf_to_images_pdf2image(document_path, dpi, poppler_path, jpeg_quality, callback)
            
        logger.error("No PDF conversion library available. Install PyMuPDF or pdf2image.")
        return {}
    else:
        logger.warning(f"Bulk conversion not implemented for {file_ext} files. Converting pages one by one.")
        return {}

def convert_pdf_to_images_fitz(pdf_path: str, dpi: int = DEFAULT_DPI, max_workers: int = DEFAULT_MAX_WORKERS, jpeg_quality: int = DEFAULT_JPEG_QUALITY, callback: Optional[Callable[[int], None]] = None) -> Dict[int, str]:
    """
    Convert all pages in a PDF to base64-encoded JPEG images using PyMuPDF with parallel processing.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution in dots per inch
        max_workers: Maximum number of worker processes
        jpeg_quality: JPEG compression quality (1-95)
        callback: Optional callback function to invoke after each page is processed
        
    Returns:
        Dict mapping page numbers to base64-encoded JPEG images
    """
    if not FITZ_AVAILABLE or not PIL_AVAILABLE:
        logger.warning("PyMuPDF not installed. Install with 'pip install PyMuPDF Pillow'")
        return {}
    
    try:
        # Open the PDF just to get page count
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        
        # Calculate zoom factor based on DPI
        zoom = dpi / 72  # 72 is the default DPI for PDF
        
        result = {}
        
        # Adjust max_workers based on page count 
        actual_workers = min(max_workers, max(1, page_count))
        
        # Use parallel processing for multiple pages
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all page conversion tasks
            future_to_page = {
                executor.submit(_process_pdf_page_fitz, pdf_path, page_idx, zoom, jpeg_quality): page_idx 
                for page_idx in range(page_count)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_page):
                page_num, img_base64 = future.result()
                if img_base64:
                    result[page_num] = img_base64
                    if callback:
                        try:
                            callback(page_num)
                        except Exception:
                            pass
        
        logger.info(f"Converted {len(result)} pages from {pdf_path} to base64 JPEG images using PyMuPDF")
        return result
    
    except Exception as e:
        logger.error(f"Error converting PDF to images using PyMuPDF: {e}")
        return {}

def convert_pdf_to_images_pdf2image(pdf_path: str, dpi: int = DEFAULT_DPI, poppler_path: str = None, jpeg_quality: int = DEFAULT_JPEG_QUALITY, callback: Optional[Callable[[int], None]] = None) -> Dict[int, str]:
    """
    Convert all pages in a PDF to base64-encoded JPEG images using pdf2image.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution in dots per inch
        poppler_path: Path to poppler binaries (required for Windows)
        jpeg_quality: JPEG compression quality (1-95)
        callback: Optional callback function to invoke after each page is processed
        
    Returns:
        Dict mapping page numbers to base64-encoded JPEG images
    """
    if not PDF2IMAGE_AVAILABLE or not PIL_AVAILABLE:
        logger.error("pdf2image and Pillow libraries are required. Install with 'pip install pdf2image Pillow'")
        return {}
    
    try:
        # Use poppler_path if provided
        conversion_kwargs = {"dpi": dpi}
        if poppler_path:
            conversion_kwargs["poppler_path"] = poppler_path
            
        images = convert_from_path(pdf_path, **conversion_kwargs)
        
        result = {}
        for i, image in enumerate(images):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Store as 1-indexed
            result[i + 1] = img_base64
            if callback:
                try:
                    callback(i + 1)
                except Exception:
                    pass
        
        logger.info(f"Converted {len(result)} pages from {pdf_path} to base64 JPEG images using pdf2image")
        return result
    except Exception as e:
        logger.error(f"Error batch converting PDF pages to images using pdf2image: {e}")
        return {}
