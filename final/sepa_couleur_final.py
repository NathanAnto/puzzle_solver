import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
import json


class PuzzleRawPointsExtractor:
    def __init__(self, input_dir="piece_noir_blanc", corners_json_path="corners_output/puzzle_corners.json"):
        self.input_dir = input_dir
        self.corners_json_path = corners_json_path
        self.piece_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
        self.pieces_raw_data = []
        self.corners_data = self.load_corners_from_json()

        if not self.piece_files:
            raise ValueError("Aucune image trouvée dans le dossier 'piece_noir_blanc'.")

    def load_corners_from_json(self):
        """Charge les données des coins depuis le fichier JSON"""
        try:
            with open(self.corners_json_path, 'r') as f:
                corners_data = json.load(f)
            print(f"Coins chargés depuis {self.corners_json_path}")
            return corners_data
        except FileNotFoundError:
            print(f"Fichier {self.corners_json_path} non trouvé. Utilisation de la détection automatique.")
            return None

    def get_corners_from_json(self, filename):
        """Récupère les coins depuis le JSON pour un fichier donné"""
        if self.corners_data is None:
            return None

        # Extraire le nom de la pièce du nom de fichier (piece_0.png -> piece_0)
        piece_name = os.path.splitext(filename)[0]

        if piece_name in self.corners_data:
            corners_info = self.corners_data[piece_name]
            corners = []

            # Extraire les coordonnées des coins dans l'ordre
            for corner in corners_info['corners']:
                corners.append([corner['x'], corner['y']])

            return np.array(corners, dtype=np.float32)
        else:
            print(f"Coins non trouvés pour {piece_name} dans le JSON")
            return None

    def detect_corners(self, contour):
        """Détecte les 4 coins de la pièce (méthode de fallback)"""
        pts = contour.reshape(-1, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def extract_side_contour(self, contour, corner1, corner2, side_name):
        """Extrait la portion du contour entre deux coins pour un côté spécifique"""
        pts = contour.reshape(-1, 2)

        # Trouver les indices des coins dans le contour
        idx1 = np.argmin(np.linalg.norm(pts - corner1, axis=1))
        idx2 = np.argmin(np.linalg.norm(pts - corner2, axis=1))

        # Assurer que idx1 < idx2 pour simplifier
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
            corner1, corner2 = corner2, corner1

        # Les deux chemins possibles
        path1 = pts[idx1:idx2 + 1]  # Chemin direct (sens horaire)
        path2 = np.concatenate([pts[idx2:], pts[:idx1 + 1]])  # Chemin long (sens anti-horaire)

        # Déterminer quel chemin prendre selon le côté
        # Calculer le centre de masse de la pièce pour référence
        center = np.mean(pts, axis=0)

        # Point milieu de chaque chemin
        if len(path1) > 1:
            mid1 = path1[len(path1) // 2]
        else:
            mid1 = (corner1 + corner2) / 2

        if len(path2) > 1:
            mid2 = path2[len(path2) // 2]
        else:
            mid2 = (corner1 + corner2) / 2

        # Choisir le bon chemin selon le côté demandé
        if side_name == "haut":
            if mid1[1] < mid2[1]:
                return path1
            else:
                return path2[::-1]
        elif side_name == "bas":
            if mid1[1] > mid2[1]:
                return path1
            else:
                return path2[::-1]
        elif side_name == "gauche":
            if mid1[0] < mid2[0]:
                return path1
            else:
                return path2[::-1]
        elif side_name == "droite":
            if mid1[0] > mid2[0]:
                return path1
            else:
                return path2[::-1]

        return path1

    def extract_17_points_from_side(self, side_points):
        """Extrait exactement 17 points équidistants le long du côté"""
        if len(side_points) < 2:
            return np.zeros((17, 2))

        # Calculer les distances cumulées le long du côté
        if len(side_points) > 1:
            distances = np.cumsum(np.sqrt(np.sum(np.diff(side_points, axis=0) ** 2, axis=1)))
            distances = np.insert(distances, 0, 0)
        else:
            distances = np.array([0])

        # Normaliser entre 0 et 1
        if distances[-1] > 0:
            distances = distances / distances[-1]

        # Créer 17 points équidistants (0, 1/16, 2/16, ..., 16/16)
        target_distances = np.linspace(0, 1, 17)

        try:
            if len(np.unique(distances)) > 1:
                # Interpoler pour obtenir les coordonnées des 17 points
                interp_x = interpolate.interp1d(distances, side_points[:, 0],
                                                kind='linear', bounds_error=False,
                                                fill_value='extrapolate')(target_distances)
                interp_y = interpolate.interp1d(distances, side_points[:, 1],
                                                kind='linear', bounds_error=False,
                                                fill_value='extrapolate')(target_distances)

                # Arrondir aux pixels entiers pour avoir des coordonnées d'image valides
                points_17 = np.column_stack([np.round(interp_x).astype(int),
                                             np.round(interp_y).astype(int)])
            else:
                # Si tous les points sont identiques, créer une ligne droite
                points_17 = np.array([np.round(np.linspace(side_points[0], side_points[-1], 17)).astype(int)])

        except (ValueError, IndexError):
            # En cas d'erreur, retourner une ligne droite entre les extrémités
            points_17 = np.round(np.linspace(side_points[0], side_points[-1], 17)).astype(int)

        return points_17

    def analyze_piece_raw(self, filename):
        """Analyse une pièce et extrait les 17 points bruts de chaque côté"""
        filepath = os.path.join(self.input_dir, filename)

        # Charger l'image
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        # Obtenir les dimensions de l'image
        height, width = mask.shape

        # Trouver le contour principal
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)

        # Essayer d'abord d'obtenir les coins depuis le JSON
        corners = self.get_corners_from_json(filename)
        corners_source = "JSON"

        # Si pas de coins dans le JSON, utiliser la détection automatique
        if corners is None:
            corners = self.detect_corners(contour)
            corners_source = "détection automatique"
            print(f"Utilisation de la {corners_source} pour {filename}")
        else:
            print(f"Utilisation des coins du {corners_source} pour {filename}")

        # Extraire les 4 côtés et leurs 17 points
        sides_raw_data = {}
        side_names = ["haut", "droite", "bas", "gauche"]

        for i in range(4):
            corner1 = corners[i]
            corner2 = corners[(i + 1) % 4]

            # Extraire les points du côté
            side_points = self.extract_side_contour(contour, corner1, corner2, side_names[i])

            # Extraire les 17 points équidistants
            points_17 = self.extract_17_points_from_side(side_points)

            # Calculer la longueur du côté
            length = np.linalg.norm(corner2 - corner1)

            # S'assurer que les coordonnées sont dans les limites de l'image
            points_17[:, 0] = np.clip(points_17[:, 0], 0, width - 1)
            points_17[:, 1] = np.clip(points_17[:, 1], 0, height - 1)

            sides_raw_data[side_names[i]] = {
                'points_17': points_17,  # Les 17 points en coordonnées image
                'points_bruts_complets': side_points,  # Tous les points du côté
                'longueur': length,
                'coin_debut': corner1,
                'coin_fin': corner2
            }

        piece_raw_data = {
            'filename': filename,
            'image_dimensions': (width, height),
            'contour': contour,
            'corners': corners,
            'corners_source': corners_source,  # Information sur la source des coins
            'sides': sides_raw_data,
            'mask': mask
        }

        return piece_raw_data

    def visualize_raw_points(self, piece_data):
        """Visualise les 17 points extraits sur l'image principale uniquement"""
        if piece_data is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        title = f"Points bruts extraits - {piece_data['filename']} (coins: {piece_data['corners_source']})"
        fig.suptitle(title, fontsize=16)

        # Image originale avec tous les points
        img_color = cv2.cvtColor(piece_data['mask'], cv2.COLOR_GRAY2BGR)

        # Dessiner le contour
        cv2.drawContours(img_color, [piece_data['contour']], -1, (0, 0, 255), 2)

        # Marquer les coins avec des couleurs spéciales pour indiquer s'ils viennent du JSON
        if piece_data['corners_source'] == "JSON":
            corner_colors = [(255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255)]  # Magenta pour JSON
        else:
            corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Couleurs normales

        for idx, corner in enumerate(piece_data['corners']):
            cv2.circle(img_color, tuple(corner.astype(int)), 6, corner_colors[idx], -1)

        # Marquer tous les points des 4 côtés avec des couleurs différentes
        side_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]  # BGRA
        side_names = ["haut", "droite", "bas", "gauche"]

        for i, side_name in enumerate(side_names):
            points_17 = piece_data['sides'][side_name]['points_17']
            color = side_colors[i]

            for j, point in enumerate(points_17):
                # Points de début et fin un peu plus gros
                if j == 0 or j == 16:
                    cv2.circle(img_color, tuple(point), 2, color, -1)
                else:
                    cv2.circle(img_color, tuple(point), 1, color, -1)

        ax.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        ax.set_title("Image avec les 17 points par côté")
        ax.axis('off')

        plt.tight_layout()
        plt.show()

    def analyze_all_pieces_raw(self, visualize=True):
        """Analyse toutes les pièces et extrait les points bruts"""
        print(f"Extraction des points bruts pour {len(self.piece_files)} pièces...")

        for filename in self.piece_files:
            print(f"\nTraitement de {filename}...")
            piece_data = self.analyze_piece_raw(filename)

            if piece_data is not None:
                self.pieces_raw_data.append(piece_data)

                if visualize:
                    self.visualize_raw_points(piece_data)

                # Afficher le résumé textuel
                print(f"  Dimensions image: {piece_data['image_dimensions']}")
                print(f"  Source des coins: {piece_data['corners_source']}")
                print(f"  Côtés détectés:")
                for side_name, side_data in piece_data['sides'].items():
                    print(
                        f"    - {side_name}: {len(side_data['points_17'])} points, longueur: {side_data['longueur']:.1f}px")

        print(f"\nExtraction terminée. {len(self.pieces_raw_data)} pièces traitées avec succès.")
        return self.pieces_raw_data

    def save_raw_points_to_json(self, output_file="puzzle_points_bruts.json"):
        """Sauvegarde les points bruts dans un fichier JSON pour réutilisation"""
        save_data = []

        for piece_data in self.pieces_raw_data:
            piece_save = {
                'filename': piece_data['filename'],
                'image_dimensions': piece_data['image_dimensions'],
                'corners': piece_data['corners'].tolist(),
                'corners_source': piece_data['corners_source'],
                'sides': {}
            }

            for side_name, side_data in piece_data['sides'].items():
                piece_save['sides'][side_name] = {
                    'points_17': side_data['points_17'].tolist(),
                    'longueur': float(side_data['longueur']),
                    'coin_debut': side_data['coin_debut'].tolist(),
                    'coin_fin': side_data['coin_fin'].tolist()
                }

            save_data.append(piece_save)

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Points bruts sauvegardés dans {output_file}")

    def load_raw_points_from_json(self, input_file="puzzle_points_bruts.json"):
        """Charge les points bruts depuis un fichier JSON"""
        try:
            with open(input_file, 'r') as f:
                loaded_data = json.load(f)

            print(f"Points bruts chargés depuis {input_file}")
            return loaded_data
        except FileNotFoundError:
            print(f"Fichier {input_file} non trouvé")
            return None

    def get_side_points_for_image(self, filename, side_name):
        """Récupère les 17 points d'un côté spécifique pour une image donnée"""
        for piece_data in self.pieces_raw_data:
            if piece_data['filename'] == filename:
                if side_name in piece_data['sides']:
                    return piece_data['sides'][side_name]['points_17']
        return None

    def apply_points_to_new_image(self, new_image_path, reference_points_data):
        """Applique les points extraits à une nouvelle image (exemple d'utilisation)"""
        new_image = cv2.imread(new_image_path)
        if new_image is None:
            print(f"Impossible de charger l'image {new_image_path}")
            return None

        # Exemple : marquer les points sur la nouvelle image
        img_marked = new_image.copy()

        for piece_info in reference_points_data:
            print(f"Application des points de {piece_info['filename']}")

            for side_name, side_data in piece_info['sides'].items():
                points_17 = np.array(side_data['points_17'])

                # Marquer chaque point (ajuster selon vos besoins)
                for i, point in enumerate(points_17):
                    color = (0, 255, 0) if i in [0, 16] else (0, 0, 255)
                    cv2.circle(img_marked, tuple(point), 3, color, -1)

        return img_marked


# === Utilisation ===
if __name__ == "__main__":
    # Créer l'extracteur avec le chemin vers le fichier JSON des coins
    extractor = PuzzleRawPointsExtractor(corners_json_path="corners_output/puzzle_corners.json")

    # Analyser toutes les pièces et extraire les points bruts
    pieces_raw_data = extractor.analyze_all_pieces_raw(visualize=True)

    # Sauvegarder les points pour réutilisation ultérieure
    extractor.save_raw_points_to_json("puzzle_points_bruts.json")

    # Exemple d'accès aux données
    if pieces_raw_data:
        piece1 = pieces_raw_data[0]
        print(f"\nExemple d'accès aux points bruts:")
        print(f"Pièce: {piece1['filename']}")
        print(f"Dimensions: {piece1['image_dimensions']}")
        print(f"Source des coins: {piece1['corners_source']}")

        # Récupérer les 17 points du côté haut
        points_haut = piece1['sides']['haut']['points_17']
        print(f"17 points du côté haut:")
        for i, point in enumerate(points_haut):
            print(f"  Point {i}: ({point[0]}, {point[1]})")

        # Exemple de récupération directe
        points_directe = extractor.get_side_points_for_image(piece1['filename'], 'haut')
        print(f"\nRécupération directe identique: {np.array_equal(points_haut, points_directe)}")

    # Démonstration du chargement depuis JSON
    print("\n=== Test de sauvegarde/chargement ===")
    loaded_data = extractor.load_raw_points_from_json("puzzle_points_bruts.json")
    if loaded_data:
        print(f"Chargé {len(loaded_data)} pièces depuis le JSON")
        print(f"Première pièce: {loaded_data[0]['filename']}")