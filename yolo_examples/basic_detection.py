"""
YOLO Basic Object Detection Example
Détecte les objets dans une image ou vidéo
"""

from ultralytics import YOLO
import cv2

def detect_image(image_path='example.jpg'):
    """Détection d'objets dans une image"""
    # Charger le modèle YOLO (télécharge automatiquement si nécessaire)
    model = YOLO('yolov8n.pt')  # n=nano, s=small, m=medium, l=large, x=xlarge
    
    # Effectuer la prédiction
    results = model(image_path)
    
    # Afficher les résultats
    for result in results:
        # Afficher l'image avec les boîtes
        result.show()
        
        # Obtenir les boîtes de détection
        boxes = result.boxes
        for box in boxes:
            # Coordonnées de la boîte
            x1, y1, x2, y2 = box.xyxy[0]
            # Confiance
            confidence = box.conf[0]
            # Classe
            class_id = box.cls[0]
            class_name = model.names[int(class_id)]
            
            print(f"Objet détecté: {class_name}, Confiance: {confidence:.2f}")
    
    # Sauvegarder le résultat
    results[0].save('output_detection.jpg')
    print("Résultat sauvegardé dans 'output_detection.jpg'")

def detect_webcam():
    """Détection en temps réel depuis la webcam"""
    model = YOLO('yolov8n.pt')
    
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prédiction
        results = model(frame)
        
        # Afficher les résultats
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO Webcam', annotated_frame)
        
        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def detect_video(video_path='example.mp4'):
    """Détection d'objets dans une vidéo"""
    model = YOLO('yolov8n.pt')
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    
    # Obtenir les propriétés de la vidéo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Créer un writer pour sauvegarder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prédiction
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # Écrire la frame
        out.write(annotated_frame)
        
        # Afficher (optionnel)
        cv2.imshow('YOLO Video', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Vidéo sauvegardée dans 'output_video.mp4'")

if __name__ == '__main__':
    print("Exemple YOLO - Détection d'objets")
    print("-" * 50)
    
    # Télécharger une image d'exemple
    print("Téléchargement d'une image d'exemple...")
    import urllib.request
    try:
        url = 'https://ultralytics.com/images/bus.jpg'
        urllib.request.urlretrieve(url, 'example.jpg')
        print("Image téléchargée!")
        
        # Exécuter la détection
        detect_image('example.jpg')
    except Exception as e:
        print(f"Erreur: {e}")
        print("Utilisez detect_webcam() pour la webcam ou fournissez votre propre image.")
