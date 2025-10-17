"""
Guide d'utilisation de l'interface graphique du neurone
"""

GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║          🧠 GUIDE D'UTILISATION - INTERFACE NEURONE 🧠          ║
╚══════════════════════════════════════════════════════════════════╝

📚 TABLE DES MATIÈRES
─────────────────────────────────────────────────────────────────

1. Démarrage rapide
2. Comprendre l'interface
3. Entraînement du neurone
4. Interprétation des résultats
5. Conseils et astuces
6. Dépannage

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ DÉMARRAGE RAPIDE
─────────────────────────────────────────────────────────────────

Pour lancer l'interface:

    cd /Users/20011409/dev/ai
    python launch_gui.py

Ou:

    cd neurone_gui
    python neurone_gui.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2️⃣ COMPRENDRE L'INTERFACE
─────────────────────────────────────────────────────────────────

L'interface est divisée en deux parties principales:

┌─ PANNEAU GAUCHE ────────────────────────────────────────────┐
│                                                              │
│  📊 Données d'entraînement                                  │
│     • OR, AND, NOR, NAND - Portes logiques pré-configurées │
│     • Données personnalisées (à venir)                      │
│                                                              │
│  ⚙️ Paramètres                                              │
│     • Taux d'apprentissage: vitesse d'apprentissage        │
│     • Époques par cycle: nombre d'itérations               │
│                                                              │
│  🎮 Contrôles                                               │
│     • ▶ Démarrer: entraînement continu                     │
│     • ⏭ Une époque: avancer pas à pas                      │
│     • ⏸ Arrêter: pause                                     │
│     • 🔄 Réinitialiser: nouveau neurone                    │
│                                                              │
│  ℹ️ Informations                                            │
│     • État actuel du neurone                                │
│     • Performance en temps réel                             │
│     • Prédictions détaillées                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘

┌─ PANNEAU DROIT (Onglets) ───────────────────────────────────┐
│                                                              │
│  🔬 Structure du Neurone                                    │
│     • Visualisation graphique du neurone                    │
│     • Poids sur les connexions                              │
│     • Couleur = signe (vert/rouge)                          │
│     • Épaisseur = magnitude                                 │
│                                                              │
│  📊 Apprentissage                                           │
│     • Graphique 1: Erreur moyenne                           │
│     • Graphique 2: Évolution des poids                      │
│     • Graphique 3: Évolution du biais                       │
│     • Graphique 4: Erreur en échelle log                    │
│                                                              │
│  🎯 Prédictions                                             │
│     • Sorties du neurone pour chaque exemple                │
│     • Comparaison avec les cibles                           │
│     • Cercle vert = correct, rouge = incorrect              │
│     • Ligne jaune = seuil de décision (0.5)                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3️⃣ ENTRAÎNEMENT DU NEURONE
─────────────────────────────────────────────────────────────────

Étapes recommandées:

1. Choisir un dataset
   └─> Commencez avec "OR" (le plus simple)

2. Observer l'état initial
   └─> Onglet "Structure" pour voir les poids aléatoires
   └─> Onglet "Prédictions" pour voir les sorties initiales

3. Configurer les paramètres
   └─> Taux d'apprentissage: 0.1 à 0.5 pour commencer
   └─> Époques: 100 pour voir l'évolution

4. Entraîner le neurone
   
   Option A - Apprentissage continu:
   └─> Cliquez "▶ Démarrer l'entraînement"
   └─> Observez l'évolution en temps réel
   └─> Cliquez "⏸ Arrêter" quand satisfait
   
   Option B - Pas à pas:
   └─> Cliquez "⏭ Une époque" plusieurs fois
   └─> Permet de voir chaque modification
   └─> Idéal pour comprendre le processus

5. Analyser les résultats
   └─> Vérifiez la précision dans "Informations"
   └─> Observez les courbes dans "Apprentissage"
   └─> Validez les prédictions dans "Prédictions"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4️⃣ INTERPRÉTATION DES RÉSULTATS
─────────────────────────────────────────────────────────────────

📊 Onglet "Structure du Neurone"
──────────────────────────────────

Connexions (lignes entre entrées et neurone):
  • VERT: Poids positif (contribution positive)
  • ROUGE: Poids négatif (contribution négative)
  • ÉPAISSEUR: Magnitude du poids (plus épais = plus important)
  • TRANSPARENCE: Intensité du poids

Exemples d'interprétation:

  Porte OR:
  • Poids x1 = 2.5 (vert, épais)  → x1 active fortement
  • Poids x2 = 2.3 (vert, épais)  → x2 active fortement
  • Biais = -1.2                   → seuil légèrement négatif
  
  Porte AND:
  • Poids x1 = 3.0 (vert, épais)  → x1 nécessaire
  • Poids x2 = 3.0 (vert, épais)  → x2 nécessaire
  • Biais = -4.5                   → seuil élevé (les deux requis)

📈 Onglet "Apprentissage"
──────────────────────────

Courbe d'erreur:
  • Décroissante: ✅ Le neurone apprend
  • Stable proche de 0: ✅ Convergence réussie
  • Oscillante: ⚠️ Taux d'apprentissage trop élevé
  • Plateau élevé: ⚠️ Données non-linéaires (XOR impossible)

Courbes des poids:
  • Convergence vers des valeurs stables: ✅ Bon
  • Oscillations: ⚠️ Réduire le taux d'apprentissage
  • Explosion (valeurs très grandes): ❌ Problème

🎯 Onglet "Prédictions"
────────────────────────

Cercles:
  • VERT: Prédiction correcte (< 10% d'erreur)
  • ROUGE: Prédiction incorrecte
  • Au-dessus de 0.5: Sortie binaire = 1
  • En-dessous de 0.5: Sortie binaire = 0

ℹ️ Panneau "Informations"
──────────────────────────

Métriques clés:

  Erreur actuelle: 0.001234
  └─> < 0.01 = Excellent
  └─> 0.01-0.1 = Bon
  └─> 0.1-0.3 = Moyen
  └─> > 0.3 = Faible

  Précision: 100.0%
  └─> Pourcentage de prédictions correctes
  └─> 100% = Parfait (toutes correctes)
  └─> 75% = 3 sur 4 correctes
  └─> 50% = Pas mieux que le hasard

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5️⃣ CONSEILS ET ASTUCES
─────────────────────────────────────────────────────────────────

🎓 Pour apprendre:

1. Commencez simple
   └─> Porte OR (la plus facile)
   └─> Mode "Une époque" pour voir chaque étape
   └─> Observez comment les poids changent

2. Expérimentez
   └─> Testez différents taux d'apprentissage
   └─> Comparez OR, AND, NOR, NAND
   └─> Observez les différences de convergence

3. Comprenez les limites
   └─> Un seul neurone ne peut résoudre XOR
   └─> Problèmes linéairement séparables uniquement

⚡ Pour optimiser:

  Convergence lente?
  ├─> Augmenter le taux d'apprentissage (0.3 → 0.7)
  ├─> Augmenter le nombre d'époques
  └─> Vérifier que les données sont appropriées

  Oscillations?
  ├─> Réduire le taux d'apprentissage (0.5 → 0.1)
  └─> Augmenter le nombre d'époques

  Résultats instables?
  ├─> Réinitialiser le neurone (nouveau départ)
  └─> Ajuster les paramètres progressivement

🔬 Pour analyser:

  • Utilisez l'échelle log pour voir la convergence fine
  • Comparez les valeurs initiales vs finales des poids
  • Observez quels poids changent le plus
  • Notez la relation entre les poids et la fonction logique

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6️⃣ DÉPANNAGE
─────────────────────────────────────────────────────────────────

❌ L'interface ne se lance pas

  Vérifiez les dépendances:
  
    python neurone_gui/test_gui.py
  
  Si des packages manquent:
  
    pip install numpy matplotlib

❌ Graphiques vides ou erreurs

  • Cliquez d'abord sur un dataset (OR, AND, etc.)
  • Puis "🔄 Réinitialiser"
  • Ensuite lancez l'entraînement

❌ Le neurone n'apprend pas (erreur reste élevée)

  Pour OR, AND, NOR, NAND:
  • Augmentez les époques (1000+)
  • Augmentez le taux d'apprentissage (0.5+)
  • Réinitialisez (nouvelles valeurs aléatoires)

  Si le problème persiste:
  • C'est peut-être XOR (impossible avec 1 neurone)
  • Vérifiez les données

❌ L'interface est lente

  • Réduisez les époques par cycle
  • Fermez les autres applications
  • Le calcul est normal sur CPU

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 CONCEPTS THÉORIQUES
─────────────────────────────────────────────────────────────────

Le neurone artificiel:
  1. Reçoit des entrées (x1, x2, ...)
  2. Multiplie par des poids (w1, w2, ...)
  3. Ajoute un biais (b)
  4. Applique une fonction d'activation (sigmoïde)
  5. Produit une sortie (0 à 1)

Formule:
  output = sigmoid(w1*x1 + w2*x2 + ... + b)

Apprentissage (rétropropagation):
  1. Calculer l'erreur: cible - sortie
  2. Ajuster les poids proportionnellement à l'erreur
  3. Répéter jusqu'à convergence

Taux d'apprentissage:
  • Trop petit: apprentissage lent
  • Optimal: convergence rapide et stable
  • Trop grand: oscillations, instabilité

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎉 AMUSEZ-VOUS!

L'objectif est de comprendre visuellement comment un neurone 
apprend. Expérimentez, observez, et découvrez les fondamentaux 
de l'intelligence artificielle!

Pour plus d'informations, consultez:
  • README.md dans neurone_gui/
  • neurone.py pour le code du neurone
  • Documentation Python AI/ML

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

if __name__ == '__main__':
    print(GUIDE)
