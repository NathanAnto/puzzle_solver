import cv2 as cv
import numpy as np
import os


def preprocess_image(image_path, resize_dim=(1080, 1440)):
    """Charge et redimensionne l'image"""
    img = cv.imread(image_path)
    img = cv.resize(img, resize_dim, interpolation=cv.INTER_LINEAR)
    return img


def detect_pieces(img, output_folder="detected_pieces"):
    """Détecte les pièces et enregistre chaque pièce détectée comme une image individuelle"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Détection des contours avec Canny
    edges = cv.Canny(blurred, 50, 150)

    # Appliquer une morphologie pour fermer les petits trous
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # Trouver les contours
    contours, _ = cv.findContours(closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Créer un dossier pour sauvegarder les images des pièces détectées
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    piece_count = 0
    margin = 20
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)

        # Filtrer les petits objets pour éviter le bruit
        if w > 30 and h > 30:
            piece_img = img[
                        y - margin:y + h + margin,
                        x - margin:x + w + margin
                        ]  # Extraire la pièce
            piece_filename = os.path.join(output_folder, f"piece_{piece_count}.jpg")
            cv.imwrite(piece_filename, piece_img)  # Sauvegarder la pièce
            piece_count += 1

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Dessiner le rectangle

    return img, edges, closed_edges


def main(image_path):
    """Pipeline principal"""
    img = preprocess_image(image_path)
    result_img, edges, mask = detect_pieces(img)

    # Sauvegarde des résultats
    cv.imwrite('detected_pieces.jpg', result_img)

    # Affichage des résultats
    cv.imshow("Detected Pieces", result_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Exécuter le programme
if __name__ == "__main__":
    image_path = "photo/20250318_142039130_iOS.jpg"  # Remplace par ton image
    main(image_path)
