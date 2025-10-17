# 🧠 Interface Graphique pour Neurone Artificiel

Interface graphique interactive pour visualiser et entraîner un neurone artificiel en temps réel.

## ✨ Fonctionnalités

### 📊 Visualisations en temps réel

1. **Structure du Neurone** 
   - Affichage graphique des connexions
   - Poids visualisés par couleur et épaisseur
   - Valeurs des poids affichées
   - Biais affiché

2. **Courbes d'Apprentissage**
   - Évolution de l'erreur
   - Évolution des poids
   - Évolution du biais
   - Erreur en échelle logarithmique

3. **Prédictions**
   - Visualisation des sorties du neurone
   - Comparaison avec les cibles
   - Indicateur de précision

### 🎮 Contrôles Interactifs

- **Données pré-configurées**: OR, AND, NOR, NAND
- **Paramètres ajustables**:
  - Taux d'apprentissage (0.01 - 1.0)
  - Nombre d'époques par cycle
- **Modes d'entraînement**:
  - Entraînement continu
  - Entraînement par époque unique
  - Pause/Reprise
- **Réinitialisation** du neurone

## 🚀 Lancement

```bash
cd neurone_gui
python neurone_gui.py
```

Ou utilisez le launcher:

```bash
python launch_gui.py
```

## 🎯 Utilisation

1. **Choisir un dataset** (OR, AND, NOR, NAND)
2. **Ajuster les paramètres** si nécessaire
3. **Démarrer l'entraînement**:
   - Cliquez sur "▶ Démarrer l'entraînement" pour un entraînement continu
   - Cliquez sur "⏭ Une époque" pour avancer pas à pas
4. **Observer**:
   - Onglet "Structure": voir les poids évoluer
   - Onglet "Apprentissage": suivre les courbes
   - Onglet "Prédictions": vérifier les résultats

## 🎨 Interface

### Panneau de Gauche
- Sélection des données
- Paramètres d'entraînement
- Contrôles (démarrer, pause, reset)
- Informations en temps réel

### Panneau de Droite (Onglets)
- **🔬 Structure du Neurone**: Visualisation graphique
- **📊 Apprentissage**: 4 graphiques de suivi
- **🎯 Prédictions**: Résultats sur les données

## 📋 Informations Affichées

```
╔══════════════════════════════════╗
║   INFORMATIONS DU NEURONE        ║
╚══════════════════════════════════╝

Époque actuelle: 150

--- Paramètres ---
Poids: ['2.3451', '1.9823']
Biais: -1.2345

--- Performance ---
Erreur actuelle: 0.001234
Erreur initiale: 0.456789
Réduction: 99.73%

--- Prédictions ---
✓ [0,0] → 0.023 (cible: 0)
✓ [0,1] → 0.987 (cible: 1)
✓ [1,0] → 0.991 (cible: 1)
✓ [1,1] → 0.998 (cible: 1)

Précision: 100.0%
```

## 🎓 Portes Logiques Disponibles

### OR (OU)
```
[0,0] → 0
[0,1] → 1
[1,0] → 1
[1,1] → 1
```

### AND (ET)
```
[0,0] → 0
[0,1] → 0
[1,0] → 0
[1,1] → 1
```

### NOR (NON-OU)
```
[0,0] → 1
[0,1] → 0
[1,0] → 0
[1,1] → 0
```

### NAND (NON-ET)
```
[0,0] → 1
[0,1] → 1
[1,0] → 1
[1,1] → 0
```

## 🔧 Dépendances

- `tkinter` (inclus avec Python)
- `numpy`
- `matplotlib`

## 💡 Astuces

1. **Convergence lente?** 
   - Augmentez le taux d'apprentissage
   - Augmentez le nombre d'époques

2. **Oscillations?**
   - Diminuez le taux d'apprentissage

3. **Expérimentation**
   - Utilisez le mode "Une époque" pour observer chaque étape
   - Comparez différentes portes logiques
   - Observez comment les poids s'ajustent

## 🎨 Codes Couleur

- **Vert**: Poids positifs / Prédictions correctes
- **Rouge**: Poids négatifs / Prédictions incorrectes
- **Jaune**: Valeurs des poids / Seuil de décision
- **Cyan**: Biais
- **Bleu**: Neurone de sortie

## 📚 Concepts Illustrés

- **Rétropropagation**: Ajustement des poids basé sur l'erreur
- **Fonction sigmoïde**: Activation du neurone
- **Gradient descent**: Descente de gradient pour minimiser l'erreur
- **Apprentissage supervisé**: Entraînement avec exemples étiquetés

## 🚧 Améliorations Futures

- [ ] Données personnalisées via interface
- [ ] Export des résultats
- [ ] Plusieurs neurones (couches)
- [ ] Autres fonctions d'activation
- [ ] Sauvegarde/chargement de modèles

---

Créé pour visualiser et comprendre le fonctionnement d'un neurone artificiel 🧠
