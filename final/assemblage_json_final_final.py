import json
import os


class PuzzleDataMerger:
    def __init__(self, classification_file="puzzle_classification.json",
                 features_file="puzzle_all_sides_features.json",
                 colors_file="puzzle_colors.json",
                 output_file="puzzle_complete_data.json"):
        """
        Fusionne les trois fichiers JSON en un seul fichier complet

        Args:
            classification_file: Fichier contenant les classifications de genre
            features_file: Fichier contenant les aires sous courbe et points normalisés
            colors_file: Fichier contenant les couleurs des segments
            output_file: Fichier de sortie avec toutes les données fusionnées
        """
        self.classification_file = classification_file
        self.features_file = features_file
        self.colors_file = colors_file
        self.output_file = output_file

    def load_json_file(self, filepath):
        """Charge un fichier JSON"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Erreur: Fichier {filepath} non trouvé")
            return None
        except json.JSONDecodeError:
            print(f"Erreur: Impossible de décoder le fichier JSON {filepath}")
            return None

    def merge_data(self):
        """Fusionne les données des trois fichiers"""
        # Charger les fichiers
        print("Chargement des fichiers JSON...")
        classification_data = self.load_json_file(self.classification_file)
        features_data = self.load_json_file(self.features_file)
        colors_data = self.load_json_file(self.colors_file)

        if not all([classification_data, features_data, colors_data]):
            print("Erreur: Impossible de charger tous les fichiers nécessaires")
            return None

        print(f"Classification: {len(classification_data)} pièces")
        print(f"Features: {len(features_data)} pièces")
        print(f"Colors: {len(colors_data)} pièces")

        # Créer un dictionnaire pour accès rapide
        classification_dict = {item['filename']: item for item in classification_data}
        features_dict = {item['filename']: item for item in features_data}
        colors_dict = {item['filename']: item for item in colors_data}

        # Fusionner les données
        merged_data = []

        # Utiliser l'ensemble de tous les noms de fichiers
        all_filenames = set()
        all_filenames.update(classification_dict.keys())
        all_filenames.update(features_dict.keys())
        all_filenames.update(colors_dict.keys())

        print(f"\nFusion des données pour {len(all_filenames)} pièces...")

        for filename in sorted(all_filenames):
            # Créer l'entrée pour cette pièce
            piece_data = {
                'filename': filename,
                'sides': {}
            }

            # Récupérer les données de chaque source
            class_data = classification_dict.get(filename, {})
            feat_data = features_dict.get(filename, {})
            color_data = colors_dict.get(filename, {})

            # Ajouter les dimensions et le centre si disponibles
            if 'image_dimensions' in class_data:
                piece_data['image_dimensions'] = class_data['image_dimensions']
            if 'piece_center' in class_data:
                piece_data['piece_center'] = class_data['piece_center']

            # Fusionner les données pour chaque côté
            side_names = ['haut', 'droite', 'bas', 'gauche']

            for side_name in side_names:
                side_info = {}

                # Données de classification (gender)
                if class_data and 'sides' in class_data and side_name in class_data['sides']:
                    class_side = class_data['sides'][side_name]
                    side_info['gender'] = class_side.get('gender', 'unknown')
                    side_info['longueur'] = class_side.get('longueur', 0)
                    side_info['deviation_max'] = class_side.get('deviation_max', 0)
                    side_info['deviation_relative'] = class_side.get('deviation_relative', 0)
                    side_info['direction'] = class_side.get('direction', 'aucune')
                    side_info['points_17'] = class_side.get('points_17', [])

                # Données de features (aire sous courbe et points normalisés)
                if feat_data and 'sides' in feat_data and side_name in feat_data['sides']:
                    feat_side = feat_data['sides'][side_name]
                    side_info['aire_sous_courbe'] = feat_side.get('aire_sous_courbe', 0)
                    side_info['points_normalises'] = feat_side.get('points_normalises', [])
                    side_info['angle_rotation'] = feat_side.get('angle_rotation', 0)
                    # Si longueur n'est pas déjà définie, l'ajouter depuis features
                    if 'longueur' not in side_info:
                        side_info['longueur'] = feat_side.get('longueur', 0)

                # Données de couleurs (segments)
                if color_data and 'sides' in color_data and side_name in color_data['sides']:
                    color_side = color_data['sides'][side_name]
                    side_info['segments'] = color_side.get('segments', [])
                    side_info['total_segments'] = color_side.get('total_segments', 0)

                # Ajouter le côté seulement s'il contient des données
                if side_info:
                    piece_data['sides'][side_name] = side_info

            # Ajouter la pièce seulement si elle a des côtés
            if piece_data['sides']:
                merged_data.append(piece_data)

        print(f"Fusion terminée: {len(merged_data)} pièces avec données complètes")

        return merged_data

    def save_merged_data(self, merged_data):
        """Sauvegarde les données fusionnées"""
        if not merged_data:
            print("Aucune donnée à sauvegarder")
            return False

        try:
            with open(self.output_file, 'w') as f:
                json.dump(merged_data, f, indent=2)
            print(f"\nDonnées fusionnées sauvegardées dans: {self.output_file}")
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False

    def verify_merged_data(self, merged_data):
        """Vérifie l'intégrité des données fusionnées"""
        print("\n=== VÉRIFICATION DES DONNÉES ===")

        total_pieces = len(merged_data)
        pieces_with_gender = 0
        pieces_with_colors = 0
        pieces_with_features = 0

        gender_counts = {'neutre': 0, 'male': 0, 'femelle': 0, 'unknown': 0}

        for piece in merged_data:
            has_gender = True
            has_colors = True
            has_features = True

            for side_name in ['haut', 'droite', 'bas', 'gauche']:
                if side_name in piece['sides']:
                    side = piece['sides'][side_name]

                    # Vérifier le genre
                    if 'gender' in side:
                        gender = side['gender']
                        if gender in gender_counts:
                            gender_counts[gender] += 1
                    else:
                        has_gender = False

                    # Vérifier les couleurs
                    if 'segments' not in side or not side['segments']:
                        has_colors = False

                    # Vérifier les features
                    if 'aire_sous_courbe' not in side or 'points_normalises' not in side:
                        has_features = False

            if has_gender:
                pieces_with_gender += 1
            if has_colors:
                pieces_with_colors += 1
            if has_features:
                pieces_with_features += 1

        print(f"Total pièces: {total_pieces}")
        print(f"Pièces avec classification de genre: {pieces_with_gender}")
        print(f"Pièces avec données de couleur: {pieces_with_colors}")
        print(f"Pièces avec features (aire/points): {pieces_with_features}")

        print(f"\nRépartition des genres (total côtés):")
        for gender, count in gender_counts.items():
            print(f"  - {gender}: {count}")

        # Vérifier quelques pièces en détail
        if merged_data:
            print(f"\nExemple de pièce complète ({merged_data[0]['filename']}):")
            piece = merged_data[0]
            for side_name, side_data in piece['sides'].items():
                print(f"  {side_name}:")
                print(f"    - gender: {side_data.get('gender', 'N/A')}")
                print(f"    - segments: {len(side_data.get('segments', []))}")
                print(f"    - aire_sous_courbe: {side_data.get('aire_sous_courbe', 'N/A')}")
                print(f"    - points_normalises: {len(side_data.get('points_normalises', []))}")

    def run(self):
        """Exécute le processus complet de fusion"""
        print("=== FUSION DES DONNÉES DU PUZZLE ===\n")

        # Fusionner les données
        merged_data = self.merge_data()

        if merged_data:
            # Vérifier les données
            self.verify_merged_data(merged_data)

            # Sauvegarder
            success = self.save_merged_data(merged_data)

            if success:
                print("\n✓ Fusion réussie!")
                return merged_data
            else:
                print("\n✗ Échec de la sauvegarde")
                return None
        else:
            print("\n✗ Échec de la fusion")
            return None


# === Utilisation ===
if __name__ == "__main__":
    # Créer le merger
    merger = PuzzleDataMerger(
        classification_file="puzzle_classification.json",
        features_file="puzzle_all_sides_features.json",
        colors_file="puzzle_colors.json",
        output_file="puzzle_complete_data.json"
    )

    # Exécuter la fusion
    merged_data = merger.run()

    if merged_data:
        print(f"\nFichier créé: puzzle_complete_data.json")
        print(f"Vous pouvez maintenant utiliser ce fichier avec le PuzzleMatcher!")