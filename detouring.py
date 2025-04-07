import cv2 as cv
import numpy as np
import os

global_piece_counter = 0  # Compteur global

def preprocess_image(image_path, resize_dim=(1080, 1440)):
    """Charge et redimensionne l'image."""
    img = cv.imread(image_path)
    if img is None:
        print(f"Erreur lors de la lecture de l'image {image_path}")
        return None
    img = cv.resize(img, resize_dim, interpolation=cv.INTER_LINEAR)
    return img

def detect_pieces(img, output_folder="detected_pieces"):
    """
    Détecte les pièces dans l'image et enregistre chaque pièce détectée dans le dossier spécifié.
    """
    global global_piece_counter

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    margin = 20
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w > 30 and h > 30:
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, img.shape[1])
            y2 = min(y + h + margin, img.shape[0])

            piece_img = img[y1:y2, x1:x2]
            piece_filename = os.path.join(output_folder, f"piece_{global_piece_counter}.jpg")
            cv.imwrite(piece_filename, piece_img)
            global_piece_counter += 1

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return img, edges, closed_edges

def process_image(image_path):
    """Traite une image unique en détectant ses pièces et en sauvegardant le résultat."""
    img = preprocess_image(image_path)
    if img is None:
        return

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    result_img, edges, mask = detect_pieces(img, output_folder="detected_pieces")

    output_path = f"detected_{base_filename}.jpg"
    cv.imwrite(output_path, result_img)
    print(f"Traitement de {image_path} terminé. Résultat enregistré sous {output_path}")

def main():
    """Parcourt le dossier 'photo' et traite toutes les images présentes."""
    photos_folder = "photo"
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    for filename in os.listdir(photos_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(photos_folder, filename)
            process_image(image_path)

if __name__ == "__main__":
    main()
