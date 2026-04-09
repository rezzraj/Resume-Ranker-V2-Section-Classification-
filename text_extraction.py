import fitz
import pytesseract
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def extract_text(file):
    read_file = file.read()
    doc = fitz.open(stream=read_file, filetype="pdf")
    full_text = ""

    try:
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                full_text += page_text + "\n"
                continue

            pixel_map = page.get_pixmap()
            img = Image.frombytes(
                "RGB",
                [pixel_map.width, pixel_map.height],
                pixel_map.samples,
            )
            full_text += pytesseract.image_to_string(img).strip() + "\n"
    finally:
        doc.close()

    return full_text
