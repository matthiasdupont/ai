import numpy as np
import matplotlib.pyplot as plt

# Fonction d'activation sigmoïde
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dérivée de la sigmoïde (pour l'apprentissage)
def sigmoid_derivative(x):
    return x * (1 - x)

# Fonction de coût : Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    """
    Calcule l'erreur quadratique moyenne.
    
    Args:
        y_true: Valeurs cibles
        y_pred: Valeurs prédites
    
    Returns:
        L'erreur quadratique moyenne
    """
    return np.mean((y_true - y_pred) ** 2)

# Fonctions de visualisation
def plot_activation_functions():
    """
    Affiche la courbe de la fonction sigmoïde et de sa dérivée.
    """
    x = np.linspace(-10, 10, 400)
    y = sigmoid(x)
    y_deriv = sigmoid_derivative(y)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="Sigmoïde", color="blue")
    plt.plot(x, y_deriv, label="Dérivée de la sigmoïde", color="red", linestyle="--")
    plt.title("Fonction d'activation sigmoïde et sa dérivée")
    plt.xlabel("x")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training_history(errors_history, weights_history, bias_history):
    """
    Affiche l'évolution de l'erreur et des paramètres pendant l'entraînement.
    
    Args:
        errors_history: Liste des erreurs moyennes par époque
        weights_history: Liste des poids à différentes époques
        bias_history: Liste des biais à différentes époques
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Graphique 1: Évolution de l'erreur
    axes[0, 0].plot(errors_history, color='red', linewidth=2)
    axes[0, 0].set_title('Évolution de l\'erreur moyenne', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Époque')
    axes[0, 0].set_ylabel('Erreur moyenne')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Graphique 2: Évolution des poids
    weights_array = np.array(weights_history)
    for i in range(weights_array.shape[1]):
        axes[0, 1].plot(weights_array[:, i], label=f'Poids {i+1}', linewidth=2)
    axes[0, 1].set_title('Évolution des poids', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Époque')
    axes[0, 1].set_ylabel('Valeur du poids')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Graphique 3: Évolution du biais
    axes[1, 0].plot(bias_history, color='green', linewidth=2)
    axes[1, 0].set_title('Évolution du biais', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Époque')
    axes[1, 0].set_ylabel('Valeur du biais')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Graphique 4: Erreur en échelle logarithmique
    axes[1, 1].plot(errors_history, color='purple', linewidth=2)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Évolution de l\'erreur (échelle log)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Époque')
    axes[1, 1].set_ylabel('Erreur moyenne (log)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Classe du neurone
class Neurone:
    def __init__(self, n_inputs):
        # Initialisation aléatoire des poids et du biais
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand(1)
        # Conversion du biais en scalaire pour éviter les warnings
        if isinstance(self.bias, np.ndarray) and self.bias.size == 1:
            self.bias = float(self.bias[0])

    def predict(self, inputs):
        # Calcul de la sortie : somme pondérée + biais, puis sigmoïde
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return sigmoid(weighted_sum)

    def train(self, inputs, target, learning_rate=0.1):
        """
        Entraînement avec descente de gradient basée sur la fonction de coût MSE.
        
        Gradient de MSE pour un exemple:
        ∂C/∂w = -(y_true - y_pred) * sigmoid'(z) * x
        ∂C/∂b = -(y_true - y_pred) * sigmoid'(z)
        
        où z = w·x + b et sigmoid'(z) = σ(z) * (1 - σ(z))
        """
        # Prédiction (forward pass)
        output = self.predict(inputs)

        # Calcul de l'erreur
        error = target - output
        
        # Calcul du gradient : δ = error * dérivée_sigmoïde
        # Pour MSE, on a: ∂L/∂output = -2 * error / n
        # Simplifié ici avec juste -error car on normalise avec le learning_rate
        delta = error * sigmoid_derivative(output)

        # Mise à jour des poids : w = w - learning_rate * (-∂L/∂w)
        # Soit : w = w + learning_rate * δ * inputs
        self.weights += learning_rate * delta * inputs
        
        # Mise à jour du biais : b = b + learning_rate * δ
        self.bias += float(learning_rate * delta)
        
        # Retourner l'erreur absolue pour le suivi
        return abs(error)
    
    def train_batch(self, X, y, learning_rate=0.1):
        """
        Entraînement sur un batch complet avec descente de gradient.
        Calcule le gradient moyen sur tous les exemples avant de mettre à jour.
        
        EXPLICATION MATHÉMATIQUE DU GRADIENT :
        =======================================
        
        Pour un neurone : output = σ(z) où z = w·x + b et σ = sigmoïde
        Fonction de coût : C = (1/2) * (y_true - output)²
        
        Par la règle de la chaîne (chain rule) :
        ∂C/∂w = ∂C/∂output × ∂output/∂z × ∂z/∂w
        
        Détails :
        1) ∂C/∂output = -(y_true - output) = -error
        2) ∂output/∂z = σ'(z) = σ(z) × (1 - σ(z)) = output × (1 - output)
           ⚠️ C'EST ICI qu'on utilise la dérivée de la sigmoïde !
        3) ∂z/∂w = x (l'entrée)
        
        Donc : ∂C/∂w = -error × σ'(z) × x
        
        Pour le biais : ∂C/∂b = -error × σ'(z) × 1 = -error × σ'(z)
        
        La dérivée de sigmoïde est ESSENTIELLE car elle représente comment
        la sortie du neurone change quand on modifie légèrement z.
        Sans elle, on ignorerait l'effet de la fonction d'activation !
        
        Args:
            X: Matrice des entrées (n_samples, n_features)
            y: Vecteur des cibles (n_samples,)
            learning_rate: Taux d'apprentissage
        
        Returns:
            Le coût MSE moyen sur le batch
        """
        n_samples = len(X)
        
        # Initialisation des gradients
        grad_weights = np.zeros_like(self.weights)
        grad_bias = 0.0
        
        # Stockage des prédictions pour calculer le coût
        predictions = []
        
        # Calcul des gradients pour chaque exemple
        for i in range(n_samples):
            # Forward pass
            output = self.predict(X[i])
            predictions.append(output)
            
            # Calcul de l'erreur : (y_true - y_pred)
            error = y[i] - output
            
            # ⚠️ ÉTAPE CRUCIALE : Calcul du gradient local
            # delta = ∂C/∂z = ∂C/∂output × ∂output/∂z
            #               = -error × sigmoid'(output)
            # 
            # sigmoid'(output) = output × (1 - output) est la dérivée de sigmoïde
            # Elle mesure la "sensibilité" de la sortie aux changements de z
            delta = error * sigmoid_derivative(output)
            
            # Accumulation des gradients
            # ∂C/∂w = delta × x
            grad_weights += delta * X[i]
            # ∂C/∂b = delta
            grad_bias += delta
        
        # Moyenne des gradients sur le batch
        grad_weights /= n_samples
        grad_bias /= n_samples
        
        # Mise à jour des paramètres (descente de gradient)
        # θ_new = θ_old - learning_rate × ∂C/∂θ
        # Ici on a un + car delta contient déjà le signe négatif de l'erreur
        self.weights += learning_rate * grad_weights
        self.bias += float(learning_rate * grad_bias)
        
        # Calcul du coût MSE
        cost = mean_squared_error(y, np.array(predictions))
        
        return cost


# Exemple d'utilisation
if __name__ == "__main__":
    # Création d'un neurone avec 2 entrées
    neurone = Neurone(n_inputs=2)

    # Données d'entraînement : entrées et cible (ex : porte logique OR)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    # Sauvegarder les paramètres initiaux
    initial_weights = neurone.weights.copy()
    initial_bias = neurone.bias
    
    # Afficher les paramètres initiaux
    print(f"\n{'='*60}")
    print("PARAMÈTRES INITIAUX")
    print(f"{'='*60}")
    print(f"Poids initiaux: {initial_weights}")
    print(f"Biais initial: {initial_bias}")

    # Historique pour les graphiques
    errors_history = []
    weights_history = []
    bias_history = []
    
    # Entraînement avec descente de gradient par batch
    print(f"\nDébut de l'entraînement (descente de gradient par batch)...")
    
    # Saisie du nombre d'époques par l'utilisateur
    while True:
        try:
            n_epochs = int(input("\nEntrez le nombre d'époques (par défaut 15000) : ") or "15000")
            if n_epochs > 0:
                break
            else:
                print("Le nombre d'époques doit être supérieur à 0.")
        except ValueError:
            print("Veuillez entrer un nombre entier valide.")
    
    # Saisie du learning rate par l'utilisateur
    while True:
        try:
            learning_rate = float(input("Entrez le learning rate (par défaut 0.5) : ") or "0.5")
            if 0 < learning_rate <= 10:
                break
            else:
                print("Le learning rate doit être entre 0 et 10.")
        except ValueError:
            print("Veuillez entrer un nombre décimal valide.")
    
    print(f"\nConfiguration de l'entraînement:")
    print(f"  - Nombre d'époques: {n_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    
    for epoch in range(n_epochs):
        # Entraînement sur tout le batch
        cost = neurone.train_batch(X, y, learning_rate=learning_rate)
        
        # Enregistrer le coût et les paramètres
        errors_history.append(cost)
        weights_history.append(neurone.weights.copy())
        bias_history.append(neurone.bias)
        
        # Affichage périodique
        if epoch % 1000 == 0:
            print(f"Époque {epoch}/{n_epochs}, Coût MSE: {cost:.6f}")

    # Calculer le nombre total d'itérations
    total_iterations = n_epochs
    
    # Afficher les valeurs finales des poids et du biais
    print(f"\n{'='*60}")
    print("RÉSULTATS DE L'ENTRAÎNEMENT")
    print(f"{'='*60}")
    print(f"Nombre d'époques: {n_epochs}")
    print(f"Méthode: Descente de gradient par batch (vraie descente de gradient)")
    print(f"Coût MSE final: {errors_history[-1]:.6f}")
    print(f"Poids finaux: {neurone.weights}")
    print(f"Biais final: {neurone.bias}")
    
    # Test
    print(f"\n{'='*60}")
    print("PRÉDICTIONS")
    print(f"{'='*60}")
    for i, x in enumerate(X):
        sortie = neurone.predict(x)
        # Conversion en float pour éviter l'erreur de format
        if isinstance(sortie, np.ndarray):
            sortie = float(sortie)
        # Seuil pour obtenir une sortie binaire
        sortie_binaire = 1 if sortie >= 0.5 else 0
        print(f"Entrée: {x}, Sortie: {sortie:.4f}, Sortie binaire: {sortie_binaire}, Cible: {y[i]}")

    # Affichage des graphiques d'entraînement
    print("\nAffichage de l'évolution de l'entraînement...")
    plot_training_history(errors_history, weights_history, bias_history)
    
    # Affichage de la fonction d'activation et de sa dérivée
    # print("\nAffichage de la fonction sigmoïde et de sa dérivée...")
    # plot_activation_functions()

