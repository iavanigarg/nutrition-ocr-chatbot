# PyMuPDF library import kar rahe hain
# Iska short naam fitz hai
import fitz

# Files aur folders handle karne ke liye
import os


def pdf_to_images(pdf_path, output_folder):
    """
    PDF ke saare pages ko images mein convert karega.

    Example:

    uploads/menu.pdf

    ↓

    output/page_1.png
    output/page_2.png
    """

    # Agar output folder nahi hai to bana do
    os.makedirs(output_folder, exist_ok=True)

    # PDF open karo
    pdf = fitz.open(pdf_path)

    # Yahan saare image paths store honge
    image_paths = []

    # PDF ke har page par loop
    for page_number in range(len(pdf)):
        
        # Current page nikalo
        page = pdf.load_page(page_number)

        # Page ko image mein convert karo
        pix = page.get_pixmap()

        # Image ka naam
        image_path = os.path.join(
            output_folder,
            f"page_{page_number + 1}.png"
        )

        # Image save karo
        pix.save(image_path)

        # List mein add karo
        image_paths.append(image_path)

    # PDF close kar do
    pdf.close()

    # Saare image paths return karo
    return image_paths