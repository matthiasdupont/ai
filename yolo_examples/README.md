# YOLO Examples 🎯

Ce répertoire contient des exemples complets d'utilisation de YOLO (You Only Look Once) avec Ultralytics.

## 📦 Installation

Les packages nécessaires sont déjà installés:
- `ultralytics` - Bibliothèque YOLO
- `opencv-python` - Traitement vidéo
- `pillow` - Traitement d'images

## 📁 Structure

### 1. `basic_detection.py`
Détection d'objets de base dans des images, vidéos et webcam.

**Fonctions:**
- `detect_image()` - Détection dans une image
- `detect_webcam()` - Détection en temps réel depuis la webcam
- `detect_video()` - Détection dans une vidéo

**Utilisation:**
```python
python basic_detection.py
```

### 2. `training_example.py`
Entraînement d'un modèle YOLO personnalisé.

**Fonctions:**
- `train_custom_model()` - Entraîner un modèle
- `validate_model()` - Valider un modèle
- `export_model()` - Exporter vers différents formats
- `create_sample_data_yaml()` - Créer un exemple de configuration

**Utilisation:**
```python
from training_example import create_sample_data_yaml
create_sample_data_yaml()
```

### 3. `advanced_features.py`
Fonctionnalités avancées de YOLO.

**Fonctions:**
- `instance_segmentation()` - Segmentation d'instance
- `pose_estimation()` - Estimation de pose humaine
- `object_tracking()` - Suivi d'objets dans les vidéos
- `classification()` - Classification d'images
- `oriented_bounding_boxes()` - Détection avec boîtes orientées
- `batch_prediction()` - Prédiction sur plusieurs images
- `custom_inference_settings()` - Paramètres personnalisés

## 🚀 Démarrage rapide

### Détection simple
```python
from ultralytics import YOLO

# Charger le modèle
model = YOLO('yolov8n.pt')

# Prédiction
results = model('image.jpg')

# Afficher
results[0].show()
```

### Détection webcam
```python
from basic_detection import detect_webcam
detect_webcam()
```

### Segmentation
```python
from advanced_features import instance_segmentation
instance_segmentation('image.jpg')
```

## 🎯 Modèles disponibles

### Détection d'objets
- `yolov8n.pt` - Nano (plus rapide)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - XLarge (plus précis)

### Segmentation
- `yolov8n-seg.pt`
- `yolov8s-seg.pt`
- etc.

### Pose estimation
- `yolov8n-pose.pt`
- `yolov8s-pose.pt`
- etc.

### Classification
- `yolov8n-cls.pt`
- `yolov8s-cls.pt`
- etc.

### OBB (Oriented Bounding Boxes)
- `yolov8n-obb.pt`
- `yolov8s-obb.pt`
- etc.

## 📊 Classes détectées (COCO dataset)

Les modèles pré-entraînés détectent 80 classes d'objets:
- Personnes, véhicules, animaux
- Objets du quotidien
- Nourriture, sports, etc.

Liste complète: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

## 💡 Conseils

1. **Performance**: Utilisez `yolov8n.pt` pour des tests rapides, `yolov8x.pt` pour la meilleure précision
2. **GPU**: Les modèles s'exécutent automatiquement sur GPU si disponible
3. **Webcam**: Utilisez 'q' pour quitter la fenêtre webcam
4. **Training**: Préparez vos données au format YOLO avant l'entraînement

## 📚 Documentation

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [GitHub Ultralytics](https://github.com/ultralytics/ultralytics)

## 🎓 Exemples d'utilisation

### Personnaliser les détections
```python
model = YOLO('yolov8n.pt')
results = model(
    'image.jpg',
    conf=0.5,           # Confiance minimum
    classes=[0, 2, 7],  # Seulement person, car, truck
)
```

### Sauvegarder les résultats
```python
results[0].save('output.jpg')
results[0].save_txt('output.txt')  # Sauvegarder les coordonnées
```

### Obtenir les détections
```python
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    confidence = box.conf[0]
    class_id = box.cls[0]
    print(f"Boîte: ({x1}, {y1}, {x2}, {y2}), Confiance: {confidence}")
```

## 🔧 Troubleshooting

- **Webcam ne fonctionne pas**: Vérifiez les permissions caméra
- **GPU non détecté**: Installez `torch` avec support CUDA
- **Modèle lent**: Utilisez un modèle plus petit (nano ou small)

Bon codage! 🚀
