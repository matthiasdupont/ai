import numpy as np
import matplotlib.pyplot as plt

# Fonction d'activation sigmoïde
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dérivée de la sigmoïde (pour l'apprentissage)
def sigmoid_derivative(x):
    return x * (1 - x)

# Classe du neurone
class Neurone:
    def __init__(self, n_inputs):
        # Initialisation aléatoire des poids et du biais
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand(1)
        # Conversion du biais en scalaire pour éviter les warnings
        if isinstance(self.bias, np.ndarray) and self.bias.size == 1:
            self.bias = float(self.bias[0])


    @staticmethod
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
    
    @staticmethod
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

    def predict(self, inputs):
        # Calcul de la sortie : somme pondérée + biais, puis sigmoïde
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return sigmoid(weighted_sum)

    def train(self, inputs, target, learning_rate=0.1):
        # Prédiction
        output = self.predict(inputs)

        # Calcul de l'erreur
        error = target - output

        # Ajustement des poids et du biais
        self.weights += learning_rate * error * sigmoid_derivative(output) * inputs
        # S'assurer que le biais reste un scalaire
        self.bias += float(learning_rate * error * sigmoid_derivative(output))
        
        # Retourner l'erreur absolue pour le suivi
        return abs(error)


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
    
    # Entraînement
    print(f"\nDébut de l'entraînement...")
    n_epochs = 10000
    for epoch in range(n_epochs):
        epoch_errors = []
        for i in range(len(X)):
            error = neurone.train(X[i], y[i])
            epoch_errors.append(error)
        
        # Enregistrer l'erreur moyenne et les paramètres toutes les 10 époques
        if epoch % 10 == 0:
            avg_error = np.mean(epoch_errors)
            errors_history.append(avg_error)
            weights_history.append(neurone.weights.copy())
            bias_history.append(neurone.bias)

    # Calculer le nombre total d'itérations
    total_iterations = n_epochs * len(X)
    
    # Afficher les valeurs finales des poids et du biais
    print(f"\n{'='*60}")
    print("RÉSULTATS DE L'ENTRAÎNEMENT")
    print(f"{'='*60}")
    print(f"Nombre d'époques: {n_epochs}")
    print(f"Nombre d'exemples par époque: {len(X)}")
    print(f"Nombre total d'itérations: {total_iterations}")
    print(f"Erreur finale moyenne: {errors_history[-1]:.6f}")
    print(f"Poids finaux: {neurone.weights}")
    print(f"Biais final: {neurone.bias}")
    
    # Test
    print(f"\n{'='*60}")
    print("PRÉDICTIONS")
    print(f"{'='*60}")
    for x in X:
        sortie = neurone.predict(x)
        # Conversion en float pour éviter l'erreur de format
        if isinstance(sortie, np.ndarray):
            sortie = float(sortie)
        # Seuil pour obtenir une sortie binaire
        sortie_binaire = 1 if sortie >= 0.5 else 0
        print(f"Entrée: {x}, Sortie: {sortie:.4f}, Sortie binaire: {sortie_binaire}")

    # Affichage des graphiques d'entraînement
    print("\nAffichage de l'évolution de l'entraînement...")
    Neurone.plot_training_history(errors_history, weights_history, bias_history)
    
    # Affichage de la fonction d'activation et de sa dérivée
    print("\nAffichage de la fonction sigmoïde et de sa dérivée...")
    Neurone.plot_activation_functions()
