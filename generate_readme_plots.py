#!/usr/bin/env python3
"""
Script pour générer des graphiques haute qualité pour le README
Génère des images PNG qui peuvent être incluses dans readme.neurone.md
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Configuration pour de beaux graphiques
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def create_sigmoid_plot():
    """Génère le graphique de la fonction sigmoïde et sa dérivée"""
    z = np.linspace(-6, 6, 200)
    sigmoid = 1 / (1 + np.exp(-z))
    sigmoid_deriv = sigmoid * (1 - sigmoid)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tracer sigmoïde
    ax.plot(z, sigmoid, 'b-', linewidth=3, label='σ(z) = 1/(1+e⁻ᶻ)', zorder=3)
    
    # Tracer dérivée
    ax.plot(z, sigmoid_deriv, 'r--', linewidth=2.5, label="σ'(z) = σ(z)·(1-σ(z))", zorder=3)
    
    # Points importants
    ax.plot(0, 0.5, 'bo', markersize=10, zorder=4, label='Point d\'inflexion (0, 0.5)')
    ax.plot(0, 0.25, 'ro', markersize=10, zorder=4, label='Maximum dérivée (0, 0.25)')
    
    # Lignes de référence
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Annotations
    ax.annotate('Saturation\n(gradient ≈ 0)', xy=(-5, 0.01), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    ax.annotate('Saturation\n(gradient ≈ 0)', xy=(3.5, 0.99), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    ax.annotate('Zone active\n(gradient max)', xy=(0.5, 0.5), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
    
    # Configuration des axes
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('z (somme pondérée)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valeur', fontsize=12, fontweight='bold')
    ax.set_title('Fonction Sigmoïde et sa Dérivée\n(Fonction d\'activation du neurone)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    plt.savefig('images/sigmoid_function.png', bbox_inches='tight', dpi=300)
    print("✓ Graphique sigmoïde sauvegardé: images/sigmoid_function.png")
    plt.close()

def create_gradient_descent_plot():
    """Génère le graphique de la descente de gradient"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Fonction de coût simple (parabole)
    w = np.linspace(-3, 5, 200)
    cost = (w - 2)**2 + 1
    
    # Sous-graphique 1: Learning rate optimal
    ax = axes[0]
    ax.plot(w, cost, 'b-', linewidth=2, alpha=0.7)
    
    # Trajectoire de descente
    w_path = [4.5, 3.7, 3.0, 2.5, 2.2, 2.05, 2.0]
    cost_path = [(w_i - 2)**2 + 1 for w_i in w_path]
    
    ax.plot(w_path, cost_path, 'ro-', markersize=8, linewidth=2, label='Trajectoire')
    ax.plot(w_path[0], cost_path[0], 'go', markersize=12, label='Départ', zorder=5)
    ax.plot(w_path[-1], cost_path[-1], 'r*', markersize=20, label='Minimum', zorder=5)
    
    # Flèches
    for i in range(len(w_path)-1):
        ax.annotate('', xy=(w_path[i+1], cost_path[i+1]), 
                    xytext=(w_path[i], cost_path[i]),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
    
    ax.set_xlabel('w (poids)', fontsize=11, fontweight='bold')
    ax.set_ylabel('C(w) (coût)', fontsize=11, fontweight='bold')
    ax.set_title('Learning Rate Optimal\nη = 0.5', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Sous-graphique 2: Learning rate trop petit
    ax = axes[1]
    ax.plot(w, cost, 'b-', linewidth=2, alpha=0.7)
    
    w_path_slow = [4.5, 4.3, 4.1, 3.9, 3.7, 3.5, 3.3, 3.1, 2.9, 2.7]
    cost_path_slow = [(w_i - 2)**2 + 1 for w_i in w_path_slow]
    
    ax.plot(w_path_slow, cost_path_slow, 'mo-', markersize=6, linewidth=2, label='Trajectoire lente')
    ax.plot(w_path_slow[0], cost_path_slow[0], 'go', markersize=12, label='Départ')
    
    ax.set_xlabel('w (poids)', fontsize=11, fontweight='bold')
    ax.set_ylabel('C(w) (coût)', fontsize=11, fontweight='bold')
    ax.set_title('Learning Rate Trop Petit\nη = 0.05', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Sous-graphique 3: Learning rate trop grand
    ax = axes[2]
    ax.plot(w, cost, 'b-', linewidth=2, alpha=0.7)
    
    w_path_osc = [4.5, 0.5, 3.5, 0.8, 3.2, 1.0, 2.9, 1.3]
    cost_path_osc = [(w_i - 2)**2 + 1 for w_i in w_path_osc]
    
    ax.plot(w_path_osc, cost_path_osc, 'co-', markersize=6, linewidth=2, label='Oscillations')
    ax.plot(w_path_osc[0], cost_path_osc[0], 'go', markersize=12, label='Départ')
    
    ax.set_xlabel('w (poids)', fontsize=11, fontweight='bold')
    ax.set_ylabel('C(w) (coût)', fontsize=11, fontweight='bold')
    ax.set_title('Learning Rate Trop Grand\nη = 2.0', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Impact du Learning Rate sur la Descente de Gradient', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('images/gradient_descent.png', bbox_inches='tight', dpi=300)
    print("✓ Graphique descente de gradient sauvegardé: images/gradient_descent.png")
    plt.close()

def create_decision_boundary_plot():
    """Génère le graphique de la frontière de décision pour OR"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Données OR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    
    # Sous-graphique 1: Avant entraînement
    ax = axes[0]
    for i in range(len(X)):
        if y[i] == 0:
            ax.plot(X[i, 0], X[i, 1], 'ro', markersize=15, label='Classe 0' if i == 0 else '')
        else:
            ax.plot(X[i, 0], X[i, 1], 'bs', markersize=15, label='Classe 1' if i == 1 else '')
    
    # Frontière aléatoire initiale
    x_line = np.linspace(-0.5, 1.5, 100)
    y_line = 0.8 * x_line + 0.3
    ax.plot(x_line, y_line, 'g--', linewidth=2, alpha=0.7, label='Frontière initiale (aléatoire)')
    
    ax.set_xlabel('x₁', fontsize=12, fontweight='bold')
    ax.set_ylabel('x₂', fontsize=12, fontweight='bold')
    ax.set_title('AVANT Entraînement\n(Poids aléatoires)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    
    # Sous-graphique 2: Après entraînement
    ax = axes[1]
    for i in range(len(X)):
        if y[i] == 0:
            ax.plot(X[i, 0], X[i, 1], 'ro', markersize=15, label='Classe 0' if i == 0 else '')
        else:
            ax.plot(X[i, 0], X[i, 1], 'bs', markersize=15, label='Classe 1' if i == 1 else '')
    
    # Frontière apprise (typique pour OR)
    # w1=6, w2=6, b=-3  =>  6*x1 + 6*x2 - 3 = 0  =>  x2 = -x1 + 0.5
    x_line = np.linspace(-0.5, 1.5, 100)
    y_line = -x_line + 0.5
    ax.plot(x_line, y_line, 'g-', linewidth=3, alpha=0.8, label='Frontière apprise')
    
    # Zone de décision
    xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 100), np.linspace(-0.3, 1.3, 100))
    Z = (6*xx + 6*yy - 3)
    Z_sigmoid = 1 / (1 + np.exp(-Z))
    
    contour = ax.contourf(xx, yy, Z_sigmoid, levels=[0, 0.5, 1], colors=['#ffcccc', '#ccccff'], alpha=0.3)
    
    ax.text(0.15, 0.15, 'Région\nClasse 0', fontsize=10, ha='center', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.7, 0.85, 'Région\nClasse 1', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('x₁', fontsize=12, fontweight='bold')
    ax.set_ylabel('x₂', fontsize=12, fontweight='bold')
    ax.set_title('APRÈS Entraînement\n(Porte logique OR)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    
    plt.suptitle('Apprentissage de la Frontière de Décision', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('images/decision_boundary.png', bbox_inches='tight', dpi=300)
    print("✓ Graphique frontière de décision sauvegardé: images/decision_boundary.png")
    plt.close()

def create_backpropagation_diagram():
    """Génère un diagramme détaillé de la rétropropagation"""
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Forward pass
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 2)
    
    # Boxes pour forward pass
    boxes_forward = [
        (1, 1, 'Entrées\nx'),
        (2.5, 1, 'Somme\nz = w·x + b'),
        (4.5, 1, 'Activation\nσ(z)'),
        (6.5, 1, 'Sortie\nŷ'),
        (8.5, 1, 'Coût\nC = ½(y-ŷ)²')
    ]
    
    for x, y, text in boxes_forward:
        rect = mpatches.FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                                       boxstyle="round,pad=0.05",
                                       edgecolor='blue', facecolor='lightblue',
                                       linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Flèches forward
    for i in range(len(boxes_forward)-1):
        x1, _, _ = boxes_forward[i]
        x2, _, _ = boxes_forward[i+1]
        ax1.annotate('', xy=(x2-0.4, 1), xytext=(x1+0.4, 1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    ax1.text(5, 1.7, 'FORWARD PASS (Propagation avant)', 
             ha='center', fontsize=14, fontweight='bold', color='blue')
    
    # Backward pass
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 2)
    
    # Boxes pour backward pass
    boxes_backward = [
        (8.5, 1, '∂C/∂ŷ\n= -(y-ŷ)'),
        (6.5, 1, '∂C/∂z\n= ∂C/∂ŷ·σ\'(z)'),
        (4.5, 1, '∂C/∂w\n= ∂C/∂z·x'),
        (2.5, 1, '∂C/∂b\n= ∂C/∂z'),
        (1, 1, 'Mise à jour\nw, b')
    ]
    
    for x, y, text in boxes_backward:
        rect = mpatches.FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                                       boxstyle="round,pad=0.05",
                                       edgecolor='red', facecolor='#ffcccc',
                                       linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Flèches backward
    for i in range(len(boxes_backward)-1):
        x1, _, _ = boxes_backward[i]
        x2, _, _ = boxes_backward[i+1]
        ax2.annotate('', xy=(x2+0.4, 1), xytext=(x1-0.4, 1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    ax2.text(5, 1.7, 'BACKWARD PASS (Rétropropagation)', 
             ha='center', fontsize=14, fontweight='bold', color='red')
    
    # Note importante
    fig.text(0.5, 0.02, 
             "⚠️ La dérivée de la sigmoïde σ'(z) est ESSENTIELLE pour calculer ∂C/∂z",
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Rétropropagation : Calcul des Gradients par la Règle de la Chaîne',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('images/backpropagation.png', bbox_inches='tight', dpi=300)
    print("✓ Diagramme rétropropagation sauvegardé: images/backpropagation.png")
    plt.close()

def main():
    """Génère tous les graphiques"""
    import os
    
    # Créer le dossier images s'il n'existe pas
    os.makedirs('images', exist_ok=True)
    
    print("🎨 Génération des graphiques pour le README...\n")
    
    create_sigmoid_plot()
    create_gradient_descent_plot()
    create_decision_boundary_plot()
    create_backpropagation_diagram()
    
    print("\n✨ Tous les graphiques ont été générés avec succès !")
    print("\n📝 Pour les inclure dans readme.neurone.md, ajoutez :")
    print("   ![Fonction Sigmoïde](images/sigmoid_function.png)")
    print("   ![Descente de Gradient](images/gradient_descent.png)")
    print("   ![Frontière de Décision](images/decision_boundary.png)")
    print("   ![Rétropropagation](images/backpropagation.png)")

if __name__ == "__main__":
    main()
