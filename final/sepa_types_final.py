import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
import json


class PuzzleGenderClassifier:
    def __init__(self, input_dir="piece_noir_blanc", corners_json_path="corners_output/puzzle_corners.json"):
        self.input_dir = input_dir
        self.corners_json_path = corners_json_path
        self.piece_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
        self.pieces_classified_data = []

        # Charger les coins depuis le fichier JSON
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
            print(
                f"Attention: Fichier {self.corners_json_path} non trouvé. Utilisation de la détection automatique des coins.")
            return {}
        except json.JSONDecodeError:
            print(
                f"Erreur lors du décodage du fichier JSON {self.corners_json_path}. Utilisation de la détection automatique des coins.")
            return {}

    def get_corners_from_json(self, filename):
        """Récupère les coins d'une pièce depuis les données JSON"""
        # Extraire le nom de base du fichier (sans extension)
        base_name = os.path.splitext(filename)[0]

        # Chercher la pièce dans les données JSON
        if base_name in self.corners_data:
            piece_data = self.corners_data[base_name]
            corners = []

            # Trier les coins par corner_id pour s'assurer de l'ordre correct
            sorted_corners = sorted(piece_data['corners'], key=lambda x: x['corner_id'])

            for corner in sorted_corners:
                corners.append([corner['x'], corner['y']])

            return np.array(corners, dtype=np.float32)

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

        # Créer 17 points équidistants
        target_distances = np.linspace(0, 1, 17)

        try:
            if len(np.unique(distances)) > 1:
                interp_x = interpolate.interp1d(distances, side_points[:, 0],
                                                kind='linear', bounds_error=False,
                                                fill_value='extrapolate')(target_distances)
                interp_y = interpolate.interp1d(distances, side_points[:, 1],
                                                kind='linear', bounds_error=False,
                                                fill_value='extrapolate')(target_distances)

                points_17 = np.column_stack([interp_x, interp_y])
            else:
                points_17 = np.linspace(side_points[0], side_points[-1], 17)

        except (ValueError, IndexError):
            points_17 = np.linspace(side_points[0], side_points[-1], 17)

        return points_17

    def classify_side_gender(self, side_points, side_name, piece_center, threshold_neutral=0.15):
        """
        Classifie le genre d'un côté: neutre, mâle ou femelle

        Args:
            side_points: points du côté (17 points)
            side_name: nom du côté (haut, bas, gauche, droite)
            piece_center: centre de la pièce
            threshold_neutral: seuil pour considérer un côté comme neutre (relatif à la longueur)

        Returns:
            dict: informations sur la classification
        """
        if len(side_points) < 3:
            return {
                'gender': 'neutre',
                'deviation_max': 0,
                'deviation_relative': 0,
                'direction': 'aucune'
            }

        # Points de début et fin
        start_point = side_points[0]
        end_point = side_points[-1]

        # Longueur du côté (ligne droite entre les extrémités)
        side_length = np.linalg.norm(end_point - start_point)

        if side_length == 0:
            return {
                'gender': 'neutre',
                'deviation_max': 0,
                'deviation_relative': 0,
                'direction': 'aucune'
            }

        # Vecteur de la ligne droite du côté
        side_vector = end_point - start_point
        side_vector_normalized = side_vector / side_length

        # Vecteur perpendiculaire pointant vers l'extérieur de la pièce
        if side_name == "haut":
            outward_vector = np.array([0, -1])  # Vers le haut
        elif side_name == "bas":
            outward_vector = np.array([0, 1])  # Vers le bas
        elif side_name == "gauche":
            outward_vector = np.array([-1, 0])  # Vers la gauche
        elif side_name == "droite":
            outward_vector = np.array([1, 0])  # Vers la droite
        else:
            outward_vector = np.array([0, 0])

        # Calculer les déviations perpendiculaires pour chaque point
        deviations = []
        for point in side_points:
            # Projection du point sur la ligne droite du côté
            point_vector = point - start_point
            projection_length = np.dot(point_vector, side_vector_normalized)
            projection_point = start_point + projection_length * side_vector_normalized

            # Vecteur de déviation
            deviation_vector = point - projection_point

            # Déviation signée (positive = vers l'extérieur, négative = vers l'intérieur)
            deviation_signed = np.dot(deviation_vector, outward_vector)
            deviations.append(deviation_signed)

        deviations = np.array(deviations)

        # Analyse des déviations
        max_deviation = np.max(np.abs(deviations))
        deviation_relative = max_deviation / side_length

        # Point de déviation maximale
        max_dev_idx = np.argmax(np.abs(deviations))
        max_dev_signed = deviations[max_dev_idx]

        # Classification
        if deviation_relative < threshold_neutral:
            gender = "neutre"
            direction = "aucune"
        elif max_dev_signed > 0:
            gender = "male"  # Tenon vers l'extérieur
            direction = "exterieur"
        else:
            gender = "femelle"  # Mortaise vers l'intérieur
            direction = "interieur"

        return {
            'gender': gender,
            'deviation_max': max_deviation,
            'deviation_relative': deviation_relative,
            'direction': direction,
            'deviations_array': deviations.tolist(),
            'max_dev_point_idx': int(max_dev_idx),
            'threshold_used': threshold_neutral
        }

    def analyze_piece_with_gender(self, filename):
        """Analyse une pièce complète avec classification de genre"""
        filepath = os.path.join(self.input_dir, filename)

        # Charger l'image
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        height, width = mask.shape

        # Trouver le contour principal
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)

        # Calculer le centre de la pièce
        M = cv2.moments(contour)
        if M["m00"] != 0:
            piece_center = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        else:
            piece_center = np.mean(contour.reshape(-1, 2), axis=0)

        # Récupérer les coins depuis le JSON ou les détecter automatiquement
        corners = self.get_corners_from_json(filename)
        corners_source = "JSON"

        if corners is None:
            print(f"  Coins non trouvés dans le JSON pour {filename}, détection automatique...")
            corners = self.detect_corners(contour)
            corners_source = "automatique"
        else:
            print(f"  Coins récupérés depuis le JSON pour {filename}")

        # Analyser les 4 côtés
        sides_data = {}
        side_names = ["haut", "droite", "bas", "gauche"]

        for i in range(4):
            corner1 = corners[i]
            corner2 = corners[(i + 1) % 4]

            # Extraire les points du côté
            side_points_raw = self.extract_side_contour(contour, corner1, corner2, side_names[i])

            # Extraire les 17 points équidistants
            points_17 = self.extract_17_points_from_side(side_points_raw)

            # Calculer la longueur du côté
            length = np.linalg.norm(corner2 - corner1)

            # Classifier le genre du côté
            gender_info = self.classify_side_gender(points_17, side_names[i], piece_center)

            # S'assurer que les coordonnées sont dans les limites de l'image
            points_17[:, 0] = np.clip(points_17[:, 0], 0, width - 1)
            points_17[:, 1] = np.clip(points_17[:, 1], 0, height - 1)

            sides_data[side_names[i]] = {
                'points_17': points_17,
                'points_bruts_complets': side_points_raw,
                'longueur': length,
                'coin_debut': corner1,
                'coin_fin': corner2,
                'gender': gender_info['gender'],
                'deviation_max': gender_info['deviation_max'],
                'deviation_relative': gender_info['deviation_relative'],
                'direction': gender_info['direction'],
                'classification_details': gender_info
            }

        piece_data = {
            'filename': filename,
            'image_dimensions': (width, height),
            'contour': contour,
            'corners': corners,
            'corners_source': corners_source,
            'piece_center': piece_center,
            'sides': sides_data,
            'mask': mask
        }

        return piece_data

    def visualize_gender_classification(self, piece_data):
        """Visualise la classification de genre des côtés"""
        if piece_data is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Couleurs pour les genres
        gender_colors = {
            'neutre': (128, 128, 128),  # Gris
            'male': (0, 0, 255),  # Rouge (BGR)
            'femelle': (255, 0, 0)  # Bleu (BGR)
        }

        # Image originale
        img_color = cv2.cvtColor(piece_data['mask'], cv2.COLOR_GRAY2BGR)

        # Dessiner le contour
        cv2.drawContours(img_color, [piece_data['contour']], -1, (0, 255, 0), 2)

        # Marquer le centre de la pièce
        center = piece_data['piece_center'].astype(int)
        cv2.circle(img_color, tuple(center), 8, (255, 255, 255), -1)
        cv2.circle(img_color, tuple(center), 6, (0, 0, 0), -1)

        # Marquer les coins avec des couleurs différentes selon la source
        corner_colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 255, 128)]
        corner_size = 8 if piece_data['corners_source'] == 'JSON' else 5

        for idx, corner in enumerate(piece_data['corners']):
            cv2.circle(img_color, tuple(corner.astype(int)), corner_size, corner_colors[idx], -1)

        # Marquer les points des côtés avec couleurs selon le genre
        side_names = ["haut", "droite", "bas", "gauche"]

        title_parts = []
        for side_name in side_names:
            side_data = piece_data['sides'][side_name]
            points_17 = side_data['points_17']
            gender = side_data['gender']
            color = gender_colors[gender]

            # Marquer tous les points du côté
            for j, point in enumerate(points_17):
                radius = 2 if j in [0, 16] else 1
                cv2.circle(img_color, tuple(point.astype(int)), radius, color, -1)

            # Marquer spécialement le point de déviation maximale pour les non-neutres
            if gender != 'neutre':
                max_dev_idx = side_data['classification_details']['max_dev_point_idx']
                max_point = points_17[max_dev_idx]
                cv2.circle(img_color, tuple(max_point.astype(int)), 4, (0, 255, 255), 2)  # Cercle jaune

            # Ajouter au titre
            deviation_pct = side_data['deviation_relative'] * 100
            title_parts.append(f"{side_name}: {gender} ({deviation_pct:.1f}%)")

        ax.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        corners_info = f"Coins: {piece_data['corners_source']}"
        ax.set_title(f"{piece_data['filename']} ({corners_info})\n" + " | ".join(title_parts), fontsize=10)
        ax.axis('off')

        # Légende
        legend_text = "Couleurs: Gris=Neutre, Rouge=Mâle, Bleu=Femelle\n"
        legend_text += "Cercle blanc/noir=Centre, Cercle jaune=Déviation max\n"
        legend_text += f"Coins: {corner_size}px = {piece_data['corners_source']}"
        ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=9, verticalalignment='bottom')

        plt.tight_layout()
        plt.show()

    def analyze_all_pieces_with_gender(self, visualize=True):
        """Analyse toutes les pièces avec classification de genre"""
        print(f"Classification de genre pour {len(self.piece_files)} pièces...")

        for filename in self.piece_files:
            print(f"\nTraitement de {filename}...")
            piece_data = self.analyze_piece_with_gender(filename)

            if piece_data is not None:
                self.pieces_classified_data.append(piece_data)

                if visualize:
                    self.visualize_gender_classification(piece_data)

                # Afficher le résumé textuel
                print(f"  Classification des côtés (coins: {piece_data['corners_source']}):")
                for side_name, side_data in piece_data['sides'].items():
                    deviation_pct = side_data['deviation_relative'] * 100
                    print(f"    - {side_name}: {side_data['gender']} "
                          f"(déviation: {deviation_pct:.1f}%, longueur: {side_data['longueur']:.1f}px)")

        print(f"\nClassification terminée. {len(self.pieces_classified_data)} pièces traitées.")
        return self.pieces_classified_data

    def save_classification_to_json(self, output_file="puzzle_classification.json"):
        """Sauvegarde la classification dans un fichier JSON"""
        save_data = []

        for piece_data in self.pieces_classified_data:
            piece_save = {
                'filename': piece_data['filename'],
                'image_dimensions': piece_data['image_dimensions'],
                'piece_center': piece_data['piece_center'].tolist(),
                'corners_source': piece_data['corners_source'],
                'sides': {}
            }

            for side_name, side_data in piece_data['sides'].items():
                piece_save['sides'][side_name] = {
                    'gender': side_data['gender'],
                    'longueur': float(side_data['longueur']),
                    'deviation_max': float(side_data['deviation_max']),
                    'deviation_relative': float(side_data['deviation_relative']),
                    'direction': side_data['direction'],
                    'points_17': side_data['points_17'].tolist()
                }

            save_data.append(piece_save)

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Classification sauvegardée dans {output_file}")

    def get_pieces_by_side_gender(self, target_gender):
        """Retourne toutes les pièces ayant au moins un côté du genre spécifié"""
        matching_pieces = []

        for piece_data in self.pieces_classified_data:
            piece_info = {
                'filename': piece_data['filename'],
                'corners_source': piece_data['corners_source'],
                'matching_sides': []
            }

            for side_name, side_data in piece_data['sides'].items():
                if side_data['gender'] == target_gender:
                    piece_info['matching_sides'].append({
                        'side': side_name,
                        'deviation_relative': side_data['deviation_relative']
                    })

            if piece_info['matching_sides']:
                matching_pieces.append(piece_info)

        return matching_pieces

    def print_classification_summary(self):
        """Affiche un résumé de toutes les classifications"""
        if not self.pieces_classified_data:
            print("Aucune donnée de classification disponible.")
            return

        total_sides = len(self.pieces_classified_data) * 4
        gender_counts = {'neutre': 0, 'male': 0, 'femelle': 0}
        corners_sources = {'JSON': 0, 'automatique': 0}

        print(f"\n=== RÉSUMÉ DE CLASSIFICATION ===")
        print(f"Nombre de pièces analysées: {len(self.pieces_classified_data)}")
        print(f"Nombre total de côtés: {total_sides}")

        for piece_data in self.pieces_classified_data:
            corners_sources[piece_data['corners_source']] += 1
            for side_data in piece_data['sides'].values():
                gender_counts[side_data['gender']] += 1

        print(f"\nSource des coins:")
        for source, count in corners_sources.items():
            percentage = (count / len(self.pieces_classified_data)) * 100
            print(f"  - {source}: {count} pièces ({percentage:.1f}%)")

        print(f"\nRépartition des genres:")
        for gender, count in gender_counts.items():
            percentage = (count / total_sides) * 100
            print(f"  - {gender.capitalize()}: {count} côtés ({percentage:.1f}%)")


# === Utilisation ===
if __name__ == "__main__":
    # Créer le classificateur avec le fichier JSON des coins
    classifier = PuzzleGenderClassifier(
        input_dir="piece_noir_blanc",
        corners_json_path="corners_output/puzzle_corners.json"
    )

    # Analyser toutes les pièces avec classification de genre
    pieces_data = classifier.analyze_all_pieces_with_gender(visualize=True)

    # Sauvegarder la classification
    classifier.save_classification_to_json("puzzle_classification.json")

    # Afficher le résumé
    classifier.print_classification_summary()

    # Exemples de recherche
    print(f"\n=== EXEMPLES DE RECHERCHE ===")

    # Chercher les pièces avec des côtés mâles
    pieces_male = classifier.get_pieces_by_side_gender('male')
    print(f"Pièces avec côtés mâles: {len(pieces_male)}")
    for piece in pieces_male[:3]:  # Afficher les 3 premières
        print(
            f"  - {piece['filename']} (coins: {piece['corners_source']}): côtés {[s['side'] for s in piece['matching_sides']]}")

    # Chercher les pièces avec des côtés femelles
    pieces_femelle = classifier.get_pieces_by_side_gender('femelle')
    print(f"Pièces avec côtés femelles: {len(pieces_femelle)}")
    for piece in pieces_femelle[:3]:
        print(
            f"  - {piece['filename']} (coins: {piece['corners_source']}): côtés {[s['side'] for s in piece['matching_sides']]}")