"""
YOLO Quick Start - Exemple simple et rapide
"""

from ultralytics import YOLO
import numpy as np
from PIL import Image

def create_sample_image():
    """Créer une image d'exemple simple"""
    # Créer une image RGB 640x480
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Ajouter un rectangle blanc (simule un objet)
    img[100:300, 200:400] = [255, 255, 255]
    
    # Ajouter un rectangle rouge
    img[250:400, 350:550] = [255, 0, 0]
    
    # Convertir en PIL Image
    pil_img = Image.fromarray(img)
    pil_img.save('sample_image.jpg')
    print("✅ Image d'exemple créée: sample_image.jpg")
    
    return 'sample_image.jpg'

def quick_detection():
    """Détection rapide sur une image"""
    print("🚀 Démarrage de YOLO...")
    print("📥 Téléchargement du modèle (première fois seulement)...")
    
    # Charger le modèle YOLO nano (le plus léger)
    model = YOLO('yolov8n.pt')
    
    print("✅ Modèle chargé!")
    
    # Créer une image d'exemple
    image_path = create_sample_image()
    
    print(f"🔍 Détection d'objets dans {image_path}...")
    
    # Effectuer la prédiction
    results = model(image_path, verbose=False)
    
    # Afficher les résultats
    print("\n📊 Résultats de la détection:")
    print("-" * 50)
    
    result = results[0]
    boxes = result.boxes
    
    if len(boxes) > 0:
        for i, box in enumerate(boxes, 1):
            # Coordonnées
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Confiance
            confidence = float(box.conf[0])
            # Classe
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            print(f"Objet {i}: {class_name}")
            print(f"  └─ Confiance: {confidence:.1%}")
            print(f"  └─ Position: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")
    else:
        print("Aucun objet détecté avec confiance suffisante.")
        print("(L'image d'exemple est simple, c'est normal!)")
    
    # Sauvegarder le résultat
    result.save('output_detection.jpg')
    print(f"\n💾 Résultat sauvegardé: output_detection.jpg")
    
    # Informations sur le modèle
    print("\n📚 Informations:")
    print(f"  • Modèle: YOLOv8 Nano")
    print(f"  • Classes détectables: {len(model.names)}")
    print(f"  • Exemples: person, car, dog, cat, bicycle, etc.")
    
    return results

def info_modeles():
    """Afficher les informations sur les modèles disponibles"""
    print("\n🎯 Modèles YOLO disponibles:")
    print("-" * 50)
    
    models = [
        ("yolov8n.pt", "Nano", "Le plus rapide, parfait pour tests"),
        ("yolov8s.pt", "Small", "Bon équilibre vitesse/précision"),
        ("yolov8m.pt", "Medium", "Plus précis, plus lent"),
        ("yolov8l.pt", "Large", "Très précis"),
        ("yolov8x.pt", "XLarge", "Maximum de précision"),
    ]
    
    for model_file, size, description in models:
        print(f"• {model_file:15} ({size:7}) - {description}")
    
    print("\n🎨 Tâches spécialisées:")
    print("-" * 50)
    print("• yolov8n-seg.pt  - Segmentation d'instance")
    print("• yolov8n-pose.pt - Estimation de pose humaine")
    print("• yolov8n-cls.pt  - Classification d'images")
    print("• yolov8n-obb.pt  - Boîtes orientées")
    
    print("\n📦 80 classes d'objets détectables (COCO dataset):")
    print("person, bicycle, car, motorcycle, airplane, bus, train, truck,")
    print("boat, bird, cat, dog, horse, bottle, chair, couch, laptop, etc.")

def exemple_utilisation():
    """Exemple d'utilisation dans votre code"""
    code = '''
# Exemple simple d'utilisation de YOLO dans votre code

from ultralytics import YOLO

# 1. Charger le modèle
model = YOLO('yolov8n.pt')

# 2. Prédiction sur une image
results = model('votre_image.jpg')

# 3. Obtenir les détections
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]  # Coordonnées
    confidence = box.conf[0]       # Confiance
    class_id = box.cls[0]          # ID de classe
    class_name = model.names[int(class_id)]  # Nom de classe
    
    print(f"{class_name}: {confidence:.2%}")

# 4. Sauvegarder le résultat
results[0].save('output.jpg')
'''
    print("\n💻 Exemple d'utilisation dans votre code:")
    print("-" * 50)
    print(code)

if __name__ == '__main__':
    print("=" * 50)
    print("   🎯 YOLO QUICK START 🎯")
    print("=" * 50)
    
    # Exécuter la détection
    quick_detection()
    
    # Afficher les informations
    info_modeles()
    
    # Afficher l'exemple de code
    exemple_utilisation()
    
    print("\n✨ Pour plus d'exemples, consultez:")
    print("  • basic_detection.py - Détection image/vidéo/webcam")
    print("  • advanced_features.py - Segmentation, pose, tracking")
    print("  • training_example.py - Entraîner votre propre modèle")
    print("  • README.md - Documentation complète")
    print("\n" + "=" * 50)
