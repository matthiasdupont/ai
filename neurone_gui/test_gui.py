"""
Script de test pour vérifier que tous les composants fonctionnent
"""

import sys
import os

# Ajouter le chemin parent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

print("=" * 60)
print("🧪 TESTS DE L'INTERFACE GRAPHIQUE DU NEURONE")
print("=" * 60)

# Test 1: Importation de neurone.py
print("\n1. Test d'importation de neurone.py...")
try:
    from neurone import Neurone, sigmoid, sigmoid_derivative
    print("   ✅ Import réussi")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test 2: Création d'un neurone
print("\n2. Test de création du neurone...")
try:
    neurone = Neurone(n_inputs=2)
    print(f"   ✅ Neurone créé avec {len(neurone.weights)} entrées")
    print(f"      Poids: {neurone.weights}")
    print(f"      Biais: {neurone.bias}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test 3: Prédiction
print("\n3. Test de prédiction...")
try:
    import numpy as np
    result = neurone.predict(np.array([1, 1]))
    print(f"   ✅ Prédiction réussie: {result:.4f}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test 4: Entraînement
print("\n4. Test d'entraînement...")
try:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    
    initial_error = abs(y[0] - neurone.predict(X[0]))
    
    for _ in range(100):
        for i in range(len(X)):
            neurone.train(X[i], y[i])
    
    final_error = abs(y[0] - neurone.predict(X[0]))
    print(f"   ✅ Entraînement réussi")
    print(f"      Erreur initiale: {initial_error:.4f}")
    print(f"      Erreur finale: {final_error:.4f}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test 5: Importation de matplotlib
print("\n5. Test d'importation de matplotlib...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend sans affichage
    import matplotlib.pyplot as plt
    print(f"   ✅ Matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    print("      Installez avec: pip install matplotlib")
    sys.exit(1)

# Test 6: Importation de tkinter
print("\n6. Test d'importation de tkinter...")
try:
    import tkinter as tk
    print(f"   ✅ Tkinter disponible (version Tcl/Tk {tk.TclVersion})")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    print("      Tkinter devrait être inclus avec Python")
    sys.exit(1)

# Test 7: Création de la figure matplotlib
print("\n7. Test de création de graphiques...")
try:
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [1, 4, 9])
    print("   ✅ Graphique créé avec succès")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test 8: Test de l'interface (import seulement)
print("\n8. Test d'importation de l'interface...")
try:
    from neurone_gui import NeuroneGUI
    print("   ✅ Interface importée avec succès")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Résumé
print("\n" + "=" * 60)
print("✅ TOUS LES TESTS SONT PASSÉS!")
print("=" * 60)
print("\n💡 Vous pouvez maintenant lancer l'interface avec:")
print("   python launch_gui.py")
print("\nOu directement:")
print("   cd neurone_gui")
print("   python neurone_gui.py")
print("\n" + "=" * 60)
