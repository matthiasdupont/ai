"""
YOLO Training Example
Entraîner un modèle YOLO personnalisé
"""

from ultralytics import YOLO

def train_custom_model():
    """
    Entraîner un modèle YOLO personnalisé
    Nécessite un dataset au format YOLO
    """
    # Charger un modèle pré-entraîné
    model = YOLO('yolov8n.pt')
    
    # Entraîner le modèle
    # Le fichier data.yaml doit contenir:
    # - path: chemin vers le dataset
    # - train: chemin vers les images d'entraînement
    # - val: chemin vers les images de validation
    # - names: liste des classes
    results = model.train(
        data='data.yaml',  # Fichier de configuration du dataset
        epochs=100,         # Nombre d'époques
        imgsz=640,         # Taille de l'image
        batch=16,          # Taille du batch
        name='yolo_custom' # Nom de l'expérience
    )
    
    return results

def validate_model():
    """Valider un modèle entraîné"""
    model = YOLO('runs/detect/yolo_custom/weights/best.pt')
    
    # Évaluation sur le dataset de validation
    metrics = model.val()
    
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    return metrics

def export_model():
    """Exporter le modèle vers différents formats"""
    model = YOLO('runs/detect/yolo_custom/weights/best.pt')
    
    # Exporter vers ONNX
    model.export(format='onnx')
    
    # Autres formats disponibles:
    # 'torchscript', 'onnx', 'openvino', 'engine', 'coreml', 
    # 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle'

def create_sample_data_yaml():
    """Créer un exemple de fichier data.yaml"""
    yaml_content = """# Dataset configuration
path: ./dataset  # Chemin racine du dataset
train: images/train  # Chemin relatif vers les images d'entraînement
val: images/val      # Chemin relatif vers les images de validation

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light

# Structure du dataset:
# dataset/
#   ├── images/
#   │   ├── train/
#   │   │   ├── image1.jpg
#   │   │   └── image2.jpg
#   │   └── val/
#   │       ├── image3.jpg
#   │       └── image4.jpg
#   └── labels/
#       ├── train/
#       │   ├── image1.txt
#       │   └── image2.txt
#       └── val/
#           ├── image3.txt
#           └── image4.txt

# Format des labels (fichiers .txt):
# <class_id> <x_center> <y_center> <width> <height>
# Toutes les valeurs sont normalisées entre 0 et 1
"""
    
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Fichier data.yaml créé!")

if __name__ == '__main__':
    print("YOLO Training Example")
    print("-" * 50)
    print("Pour entraîner un modèle, vous devez d'abord:")
    print("1. Préparer votre dataset au format YOLO")
    print("2. Créer un fichier data.yaml")
    print("3. Lancer l'entraînement avec train_custom_model()")
    print()
    
    create_sample_data_yaml()
