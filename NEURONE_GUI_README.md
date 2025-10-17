# 🧠 Interface Graphique du Neurone - Guide de Démarrage Rapide

## ⚡ Lancement en 10 secondes

```bash
cd /Users/20011409/dev/ai
python launch_gui.py
```

## 🎯 Ce que fait cette interface

Une **visualisation interactive en temps réel** de votre neurone artificiel:

- 🔬 Structure du neurone avec poids colorés
- 📊 Graphiques d'apprentissage en direct
- 🎯 Prédictions visuelles
- 🎮 Contrôles interactifs (play, pause, step)

## 📸 Aperçu

```
┌─────────────────────────────────────────────────────────┐
│              🧠 Neurone Artificiel                      │
├──────────────┬──────────────────────────────────────────┤
│              │  [Visualisation graphique du neurone]    │
│  Données     │  • Entrées x1, x2                        │
│  OR ✓        │  • Poids en vert/rouge                   │
│  AND         │  • Connexions animées                    │
│  NOR         │                                          │
│  NAND        ├──────────────────────────────────────────┤
│              │  [Graphiques d'apprentissage]            │
│  ⚙️ Params   │  • Erreur décroissante                   │
│  LR: 0.5     │  • Poids qui s'ajustent                  │
│  Ep: 100     │  • Convergence visible                   │
│              │                                          │
│  🎮 Ctrl     ├──────────────────────────────────────────┤
│  ▶ Start     │  [Prédictions]                           │
│  ⏭ Step      │  • Cercles verts = correct               │
│  ⏸ Stop      │  • Cercles rouges = incorrect            │
│  🔄 Reset    │  • Précision affichée                    │
│              │                                          │
│  ℹ️ Info     │                                          │
│  Époque: 150 │                                          │
│  Erreur: 0.01│                                          │
│  Préc: 100%  │                                          │
└──────────────┴──────────────────────────────────────────┘
```

## 🚀 Utilisation simple

1. **Choisir une porte logique** (ex: OR)
2. **Cliquer sur "▶ Démarrer"**
3. **Observer** l'apprentissage en direct!

## 📁 Organisation

```
/Users/20011409/dev/ai/
├── neurone.py              # Votre neurone original
├── launch_gui.py           # 👈 Lancez depuis ici!
│
└── neurone_gui/            # Dossier de l'interface
    ├── neurone_gui.py      # Interface principale
    ├── start.sh            # Alternative: ./start.sh
    ├── test_gui.py         # Tests: python test_gui.py
    ├── GUIDE.py            # Guide détaillé
    ├── DOCUMENTATION_COMPLETE.md  # Doc complète
    └── README.md           # Documentation
```

## 📚 Documentation

- **Démarrage rapide**: Ce fichier
- **Guide complet**: `neurone_gui/GUIDE.py`
- **Documentation**: `neurone_gui/DOCUMENTATION_COMPLETE.md`
- **README**: `neurone_gui/README.md`

## 🎓 Exemples d'utilisation

### Exemple 1: Observer l'apprentissage

```
1. Sélectionner "OR"
2. Cliquer "▶ Démarrer"
3. Regarder:
   • Les poids devenir positifs (~2-3)
   • Le biais devenir négatif (~-1)
   • L'erreur tendre vers 0
   • Les prédictions devenir correctes
```

### Exemple 2: Mode pas à pas

```
1. Sélectionner "AND"
2. Cliquer "⏭ Une époque" plusieurs fois
3. Observer chaque modification des poids
4. Comprendre la convergence progressive
```

### Exemple 3: Comparer les portes

```
1. Entraîner OR → noter les poids
2. Entraîner AND → comparer
3. Observer: AND a un biais plus négatif
   (besoin des DEUX entrées)
```

## 🎨 Codes couleur

- 🟢 **Vert**: Poids positif / Prédiction correcte
- 🔴 **Rouge**: Poids négatif / Prédiction incorrecte
- 🟡 **Jaune**: Valeurs des poids / Seuil
- 🔵 **Cyan**: Biais
- 🟦 **Bleu**: Neurone de sortie

## ⚙️ Paramètres recommandés

| Paramètre | Débutant | Avancé |
|-----------|----------|--------|
| Taux d'apprentissage | 0.1-0.3 | 0.5-0.7 |
| Époques par cycle | 50-100 | 500-1000 |

## 🔧 Dépannage rapide

**Problème**: Interface ne démarre pas
```bash
cd neurone_gui
python test_gui.py  # Vérifie tout
```

**Problème**: Graphiques vides
```
1. Cliquer sur "OR" (ou autre porte)
2. Cliquer sur "🔄 Réinitialiser"
3. Relancer
```

**Problème**: N'apprend pas
```
• Augmenter taux d'apprentissage → 0.5
• Augmenter époques → 1000
• Cliquer "🔄 Réinitialiser"
```

## 💡 Astuces

- **Pour apprendre**: Mode "Une époque" et observer chaque étape
- **Pour tester**: Mode "Démarrer" et comparer différentes portes
- **Pour comprendre**: Regarder comment les poids se stabilisent

## 🎯 Portes logiques disponibles

| Porte | [0,0] | [0,1] | [1,0] | [1,1] | Difficulté |
|-------|-------|-------|-------|-------|------------|
| OR    | 0     | 1     | 1     | 1     | ⭐ Facile  |
| NOR   | 1     | 0     | 0     | 0     | ⭐⭐ Moyen  |
| AND   | 0     | 0     | 0     | 1     | ⭐⭐ Moyen  |
| NAND  | 1     | 1     | 1     | 0     | ⭐⭐ Moyen  |

## 📖 Concepts illustrés

- ✅ **Perceptron** (neurone simple)
- ✅ **Fonction sigmoïde** (activation)
- ✅ **Rétropropagation** (apprentissage)
- ✅ **Gradient descent** (optimisation)
- ✅ **Convergence** (stabilisation)

## 🎉 Prochaines étapes

Après avoir maîtrisé l'interface:

1. Comprendre pourquoi XOR est impossible avec 1 neurone
2. Implémenter plusieurs neurones (réseau)
3. Tester d'autres fonctions d'activation (ReLU, tanh)
4. Appliquer à des données réelles

## 📞 Aide

- Consultez `neurone_gui/GUIDE.py` pour le guide complet
- Lisez `neurone_gui/DOCUMENTATION_COMPLETE.md`
- Examinez le code commenté dans `neurone_gui.py`

---

**Créé pour visualiser et comprendre les neurones artificiels 🧠**

Bon apprentissage! 🚀
