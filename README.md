# 🧠 Projet AI - Intelligence Artificielle

Projet d'apprentissage et d'expérimentation en Intelligence Artificielle.

## 📚 Contenu du Projet

### 1️⃣ Neurone Artificiel avec Interface Graphique 🎨

**Interface graphique interactive** pour visualiser et comprendre le fonctionnement d'un neurone artificiel.

- 📁 Fichier principal: `neurone.py`
- 🖼️ Interface: `neurone_gui/`
- 📖 Documentation: `NEURONE_GUI_README.md`

**Lancement:**
```bash
python launch_gui.py
```

**Fonctionnalités:**
- ✅ Visualisation en temps réel du neurone
- ✅ Graphiques d'apprentissage interactifs
- ✅ Portes logiques pré-configurées (OR, AND, NOR, NAND)
- ✅ Contrôles interactifs (play, pause, step-by-step)
- ✅ Prédictions visuelles

### 2️⃣ YOLO - Détection d'Objets 🎯

**Exemples complets** d'utilisation de YOLO (You Only Look Once) pour la détection d'objets.

- 📁 Répertoire: `yolo_examples/`
- 📖 Documentation: `yolo_examples/README.md`

**Lancement rapide:**
```bash
cd yolo_examples
python quick_start.py
```

**Fonctionnalités:**
- ✅ Détection d'objets dans images/vidéos
- ✅ Détection en temps réel (webcam)
- ✅ Segmentation d'instance
- ✅ Estimation de pose
- ✅ Suivi d'objets (tracking)
- ✅ Entraînement de modèles personnalisés

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/matthiasdupont/ai.git
cd ai

# Créer un environnement virtuel (optionnel)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Installer les dépendances
pip install -r requirements.txt
```

## 📦 Dépendances

- `numpy` - Calculs numériques
- `matplotlib` - Visualisations
- `tkinter` - Interface graphique (inclus avec Python)
- `ultralytics` - YOLO pour la détection d'objets
- `opencv-python` - Traitement vidéo
- `pillow` - Traitement d'images
- `nltk`, `pandas`, `textblob`, `wordcloud` - Autres outils ML

## 📁 Structure du Projet

```
ai/
├── 🧠 neurone.py                   # Neurone artificiel (classe principale)
├── 🚀 launch_gui.py                # Lanceur de l'interface graphique
├── 📖 NEURONE_GUI_README.md        # Guide neurone GUI
│
├── 🎨 neurone_gui/                 # Interface graphique du neurone
│   ├── neurone_gui.py              # Application principale
│   ├── neurone_gui_demo.py         # Version démo automatique
│   ├── test_gui.py                 # Tests de validation
│   ├── start.sh                    # Script de lancement
│   ├── demo_info.py                # Informations détaillées
│   ├── GUIDE.py                    # Guide d'utilisation
│   ├── DOCUMENTATION_COMPLETE.md   # Documentation technique
│   └── README.md                   # Documentation interface
│
├── 🎯 yolo_examples/               # Exemples YOLO
│   ├── quick_start.py              # Démarrage rapide
│   ├── basic_detection.py          # Détection de base
│   ├── advanced_features.py        # Fonctionnalités avancées
│   ├── training_example.py         # Entraînement personnalisé
│   └── README.md                   # Documentation YOLO
│
├── 📋 requirements.txt             # Dépendances Python
└── 📖 README.md                    # Ce fichier
```

## 🎓 Utilisation

### Interface Graphique du Neurone

**Démarrage rapide:**
```bash
python launch_gui.py
```

**Workflow:**
1. Choisir une porte logique (OR recommandé)
2. Cliquer "▶ Démarrer l'entraînement"
3. Observer l'apprentissage en temps réel
4. Analyser les résultats dans les différents onglets

**Documentation:**
- Guide rapide: `NEURONE_GUI_README.md`
- Guide complet: `neurone_gui/GUIDE.py`
- Doc technique: `neurone_gui/DOCUMENTATION_COMPLETE.md`

### YOLO Détection d'Objets

**Démarrage rapide:**
```bash
cd yolo_examples
python quick_start.py
```

**Exemples disponibles:**
- `basic_detection.py` - Images, vidéos, webcam
- `advanced_features.py` - Segmentation, pose, tracking
- `training_example.py` - Entraîner votre modèle

**Documentation:**
- `yolo_examples/README.md`

## 💡 Concepts Illustrés

### Neurone Artificiel
- ✅ Perceptron simple
- ✅ Fonction d'activation (sigmoïde)
- ✅ Rétropropagation
- ✅ Gradient descent
- ✅ Convergence
- ✅ Portes logiques (séparabilité linéaire)

### YOLO
- ✅ Détection d'objets en temps réel
- ✅ Réseaux de neurones convolutifs (CNN)
- ✅ Segmentation d'instance
- ✅ Transfer learning
- ✅ Suivi d'objets (tracking)

## 🧪 Tests

### Tester l'interface du neurone:
```bash
cd neurone_gui
python test_gui.py
```

### Tester YOLO:
```bash
cd yolo_examples
python quick_start.py
```

## 📚 Documentation Détaillée

### Neurone
- **NEURONE_GUI_README.md** - Guide de démarrage rapide
- **neurone_gui/README.md** - Documentation de l'interface
- **neurone_gui/GUIDE.py** - Guide d'utilisation détaillé
- **neurone_gui/DOCUMENTATION_COMPLETE.md** - Documentation technique complète

### YOLO
- **yolo_examples/README.md** - Documentation complète YOLO
- Code commenté dans chaque fichier d'exemple

## 🎯 Prochaines Étapes

### Court terme
- [ ] Ajouter données personnalisées dans l'interface neurone
- [ ] Implémenter XOR avec réseau multi-couches
- [ ] Ajouter d'autres fonctions d'activation (ReLU, tanh)
- [ ] Export des résultats d'entraînement

### Moyen terme
- [ ] Réseau de neurones complet (multiple couches)
- [ ] Interface pour réseaux multi-couches
- [ ] Intégration YOLO avec interface graphique
- [ ] Datasets personnalisés pour YOLO

### Long terme
- [ ] Réseaux convolutifs (CNN)
- [ ] Réseaux récurrents (RNN/LSTM)
- [ ] Apprentissage par renforcement
- [ ] Transformers

## 🤝 Contribution

Ce projet est à but éducatif. N'hésitez pas à:
- Expérimenter avec le code
- Ajouter de nouvelles fonctionnalités
- Améliorer la documentation
- Partager vos découvertes

## 📞 Support

Pour toute question:
1. Consultez la documentation dans chaque dossier
2. Lisez les guides d'utilisation
3. Examinez le code commenté
4. Lancez les scripts de test

## 📝 Licence

Projet éducatif - Libre d'utilisation et de modification

---

**Créé avec ❤️ pour l'apprentissage de l'Intelligence Artificielle**

*Dernière mise à jour: 17 octobre 2025*