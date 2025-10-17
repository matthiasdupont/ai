"""
YOLO Advanced Features
Fonctionnalités avancées: segmentation, pose estimation, tracking
"""

from ultralytics import YOLO
import cv2

def instance_segmentation(image_path='example.jpg'):
    """
    Segmentation d'instance - détecte et segmente les objets
    """
    # Charger le modèle de segmentation
    model = YOLO('yolov8n-seg.pt')
    
    # Prédiction
    results = model(image_path)
    
    # Afficher les résultats
    for result in results:
        result.show()
        
        # Obtenir les masques de segmentation
        if result.masks is not None:
            masks = result.masks.data
            print(f"Nombre d'objets segmentés: {len(masks)}")
    
    results[0].save('output_segmentation.jpg')
    print("Segmentation sauvegardée!")

def pose_estimation(image_path='example.jpg'):
    """
    Estimation de pose - détecte les points clés du corps humain
    """
    # Charger le modèle de pose
    model = YOLO('yolov8n-pose.pt')
    
    # Prédiction
    results = model(image_path)
    
    # Afficher les résultats
    for result in results:
        result.show()
        
        # Obtenir les keypoints
        if result.keypoints is not None:
            keypoints = result.keypoints.data
            print(f"Nombre de personnes détectées: {len(keypoints)}")
    
    results[0].save('output_pose.jpg')
    print("Estimation de pose sauvegardée!")

def object_tracking(video_path='example.mp4'):
    """
    Suivi d'objets dans une vidéo
    """
    model = YOLO('yolov8n.pt')
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    
    # Tracking
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Utiliser le tracker (BoT-SORT ou ByteTrack)
        results = model.track(frame, persist=True)
        
        # Afficher les résultats avec les IDs de tracking
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO Tracking', annotated_frame)
        
        # Afficher les IDs des objets suivis
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            print(f"Objets suivis: {track_ids}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def classification(image_path='example.jpg'):
    """
    Classification d'images
    """
    # Charger le modèle de classification
    model = YOLO('yolov8n-cls.pt')
    
    # Prédiction
    results = model(image_path)
    
    # Afficher les résultats
    for result in results:
        probs = result.probs
        top5_indices = probs.top5
        top5_conf = probs.top5conf
        
        print("\nTop 5 prédictions:")
        for i, (idx, conf) in enumerate(zip(top5_indices, top5_conf), 1):
            class_name = model.names[idx]
            print(f"{i}. {class_name}: {conf:.2%}")

def oriented_bounding_boxes(image_path='example.jpg'):
    """
    Détection avec boîtes englobantes orientées (OBB)
    Utile pour les objets en rotation (images aériennes, etc.)
    """
    model = YOLO('yolov8n-obb.pt')
    
    results = model(image_path)
    
    for result in results:
        result.show()
        
        if result.obb is not None:
            obb = result.obb
            print(f"Nombre d'objets détectés avec OBB: {len(obb)}")
    
    results[0].save('output_obb.jpg')
    print("OBB sauvegardée!")

def batch_prediction(image_folder='images/'):
    """
    Prédiction par batch sur plusieurs images
    """
    model = YOLO('yolov8n.pt')
    
    # Prédire sur toutes les images d'un dossier
    results = model(image_folder, stream=True)
    
    for i, result in enumerate(results):
        print(f"\nImage {i+1}:")
        boxes = result.boxes
        for box in boxes:
            class_name = model.names[int(box.cls[0])]
            confidence = box.conf[0]
            print(f"  - {class_name}: {confidence:.2f}")

def custom_inference_settings():
    """
    Paramètres personnalisés pour l'inférence
    """
    model = YOLO('yolov8n.pt')
    
    # Prédiction avec paramètres personnalisés
    results = model(
        'example.jpg',
        conf=0.25,        # Seuil de confiance minimum
        iou=0.45,         # Seuil IoU pour NMS
        imgsz=640,        # Taille de l'image
        half=False,       # Utiliser FP16 (plus rapide sur GPU)
        device='cpu',     # 'cpu', '0' (GPU), '0,1,2,3' (multi-GPU)
        max_det=300,      # Nombre maximum de détections
        vid_stride=1,     # Stride pour les vidéos (traiter 1 frame sur N)
        classes=[0, 2],   # Filtrer par classes (0=person, 2=car)
        verbose=False     # Désactiver les logs
    )
    
    return results

if __name__ == '__main__':
    print("YOLO Advanced Features")
    print("-" * 50)
    print("\nFonctionnalités disponibles:")
    print("1. instance_segmentation() - Segmentation d'instance")
    print("2. pose_estimation() - Estimation de pose")
    print("3. object_tracking() - Suivi d'objets")
    print("4. classification() - Classification d'images")
    print("5. oriented_bounding_boxes() - Boîtes orientées")
    print("6. batch_prediction() - Prédiction par batch")
    print("7. custom_inference_settings() - Paramètres personnalisés")
    print("\nChaque fonction téléchargera le modèle nécessaire automatiquement.")
