#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemple d'utilisation de YOLO (YOLOv8)
====================================

Ce script démontre l'utilisation de YOLOv8 pour la détection d'objets.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def demo_detection_image(image_path=None):
    """
    Détection d'objets sur une image avec YOLOv8
    
    Args:
        image_path: Chemin vers l'image (si None, utilise une image de test)
    """
    print("=" * 60)
    print("DÉTECTION D'OBJETS AVEC YOLO")
    print("=" * 60)
    
    # Charger le modèle YOLOv8 pré-entraîné (nano version, plus légère)
    print("\n📥 Chargement du modèle YOLOv8...")
    model = YOLO('yolov8n.pt')  # yolov8n = nano (le plus léger)
    
    # Si aucune image n'est fournie, YOLO peut télécharger une image de test
    if image_path is None:
        print("⚠️  Aucune image fournie. Utilisation d'une image de test...")
        # YOLO téléchargera automatiquement une image de test
        image_path = 'https://ultralytics.com/images/bus.jpg'
    
    # Effectuer la détection
    print(f"\n🔍 Détection sur: {image_path}")
    results = model(image_path)
    
    # Afficher les résultats
    print("\n📊 RÉSULTATS DE LA DÉTECTION:")
    print("-" * 60)
    
    for result in results:
        # Récupérer les boîtes détectées
        boxes = result.boxes
        
        if len(boxes) == 0:
            print("Aucun objet détecté.")
        else:
            print(f"Nombre d'objets détectés: {len(boxes)}")
            print()
            
            # Afficher les détails de chaque détection
            for i, box in enumerate(boxes):
                # Classe de l'objet
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Confiance
                confidence = float(box.conf[0])
                
                # Coordonnées de la boîte (x1, y1, x2, y2)
                coords = box.xyxy[0].cpu().numpy()
                
                print(f"Objet {i+1}:")
                print(f"  - Classe: {class_name}")
                print(f"  - Confiance: {confidence:.2%}")
                print(f"  - Position: x1={coords[0]:.1f}, y1={coords[1]:.1f}, "
                      f"x2={coords[2]:.1f}, y2={coords[3]:.1f}")
                print()
        
        # Sauvegarder l'image avec les détections
        output_path = 'yolo_detection_result.jpg'
        result.save(output_path)
        print(f"✅ Image avec détections sauvegardée: {output_path}")
        
        # Afficher l'image (optionnel, nécessite un environnement graphique)
        try:
            result.show()
        except:
            print("⚠️  Impossible d'afficher l'image (pas d'environnement graphique)")

def demo_detection_webcam():
    """
    Détection d'objets en temps réel via webcam
    """
    print("=" * 60)
    print("DÉTECTION EN TEMPS RÉEL (WEBCAM)")
    print("=" * 60)
    print("Appuyez sur 'q' pour quitter")
    
    # Charger le modèle
    model = YOLO('yolov8n.pt')
    
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erreur: Impossible d'ouvrir la webcam")
        return
    
    print("\n🎥 Webcam ouverte. Détection en cours...")
    
    while True:
        # Lire une frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Effectuer la détection
        results = model(frame, verbose=False)
        
        # Dessiner les résultats sur la frame
        annotated_frame = results[0].plot()
        
        # Afficher la frame
        cv2.imshow('YOLO Detection', annotated_frame)
        
        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam fermée")

def demo_detection_video(video_path):
    """
    Détection d'objets sur une vidéo
    
    Args:
        video_path: Chemin vers la vidéo
    """
    print("=" * 60)
    print("DÉTECTION D'OBJETS SUR VIDÉO")
    print("=" * 60)
    
    # Charger le modèle
    model = YOLO('yolov8n.pt')
    
    # Vérifier que la vidéo existe
    if not Path(video_path).exists():
        print(f"❌ Erreur: Fichier vidéo non trouvé: {video_path}")
        return
    
    print(f"\n🎬 Traitement de la vidéo: {video_path}")
    
    # Effectuer la détection sur la vidéo
    results = model(video_path, stream=True)
    
    # Ouvrir la vidéo pour obtenir les propriétés
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Créer un writer pour sauvegarder la vidéo résultante
    output_path = 'yolo_video_result.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    for result in results:
        frame_count += 1
        annotated_frame = result.plot()
        out.write(annotated_frame)
        
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count} traité...")
    
    out.release()
    print(f"\n✅ Vidéo traitée: {frame_count} frames")
    print(f"✅ Vidéo avec détections sauvegardée: {output_path}")

def demo_info_modele():
    """
    Affiche les informations sur le modèle YOLO
    """
    print("=" * 60)
    print("INFORMATIONS SUR LE MODÈLE YOLO")
    print("=" * 60)
    
    # Charger le modèle
    model = YOLO('yolov8n.pt')
    
    print(f"\nModèle: YOLOv8n (nano)")
    print(f"Nombre de classes: {len(model.names)}")
    print(f"\nClasses détectables:")
    print("-" * 60)
    
    # Afficher les classes par colonnes
    classes = list(model.names.values())
    for i in range(0, len(classes), 4):
        row = classes[i:i+4]
        print("  " + ", ".join(f"{c:15s}" for c in row))
    
    print("\n" + "=" * 60)

def main():
    """
    Menu principal pour tester les différentes fonctionnalités
    """
    print("\n" + "=" * 60)
    print("EXEMPLES D'UTILISATION DE YOLO")
    print("=" * 60)
    print("\nOptions disponibles:")
    print("1. Informations sur le modèle")
    print("2. Détection sur une image de test")
    print("3. Détection sur une image personnalisée")
    print("4. Détection en temps réel (webcam)")
    print("5. Détection sur une vidéo")
    print()
    
    choix = input("Choisissez une option (1-5, ou 'q' pour quitter): ").strip()
    
    if choix == '1':
        demo_info_modele()
    
    elif choix == '2':
        demo_detection_image()
    
    elif choix == '3':
        image_path = input("Entrez le chemin de l'image: ").strip()
        demo_detection_image(image_path)
    
    elif choix == '4':
        demo_detection_webcam()
    
    elif choix == '5':
        video_path = input("Entrez le chemin de la vidéo: ").strip()
        demo_detection_video(video_path)
    
    elif choix.lower() == 'q':
        print("Au revoir!")
    
    else:
        print("Option invalide")

if __name__ == "__main__":
    # Par défaut, exécuter la détection sur une image de test
    print("Démarrage de la détection d'objets avec YOLO...")
    print("Le modèle sera téléchargé automatiquement lors de la première utilisation.\n")
    
    # Lancer le menu
    main()