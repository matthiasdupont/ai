#!/usr/bin/env python3
"""
Lanceur simplifié pour l'interface graphique du neurone
"""

import os
import sys

# Ajouter le répertoire courant (où se trouve ce script) au path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Vérifier que neurone.py existe
neurone_path = os.path.join(current_dir, 'neurone.py')
if not os.path.exists(neurone_path):
    print("❌ Erreur: neurone.py non trouvé!")
    print(f"   Chemin attendu: {neurone_path}")
    sys.exit(1)

print("🧠 Lancement de l'interface graphique du neurone...")
print("=" * 60)

try:
    # Ajouter le sous-dossier neurone_gui au path
    neurone_gui_dir = os.path.join(current_dir, 'neurone_gui')
    sys.path.insert(0, neurone_gui_dir)
    
    from neurone_gui import main
    main()
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    print("\n📦 Packages requis:")
    print("   - numpy")
    print("   - matplotlib")
    print("   - tkinter (inclus avec Python)")
    print("\n💡 Installez les dépendances avec:")
    print("   pip install numpy matplotlib")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
