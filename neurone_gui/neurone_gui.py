"""
Interface graphique pour visualiser un neurone artificiel
Affichage en temps réel de l'entraînement et de la structure du neurone
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
import sys
import os

# Importer la classe Neurone
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neurone import Neurone, sigmoid


class NeuroneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Visualisation du Neurone Artificiel")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Style
        self.setup_style()
        
        # Variables
        self.neurone = None
        self.is_training = False
        self.training_data = None
        self.current_epoch = 0
        self.errors_history = []
        self.weights_history = []
        self.bias_history = []
        
        # Interface
        self.create_widgets()
        
        # Initialiser avec une porte logique OR par défaut
        self.load_or_gate()
        
    def setup_style(self):
        """Configuration du style de l'interface"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Couleurs
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white', font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'), foreground='#4CAF50')
        style.configure('TButton', font=('Arial', 10, 'bold'))
        style.map('TButton', background=[('active', '#4CAF50')])
        
    def create_widgets(self):
        """Créer tous les widgets de l'interface"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== PARTIE GAUCHE: Contrôles =====
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Titre
        title_label = ttk.Label(left_frame, text="🧠 Neurone Artificiel", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # --- Données d'entraînement ---
        data_frame = ttk.LabelFrame(left_frame, text="Données d'entraînement", padding=10)
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(data_frame, text="Porte OR", command=self.load_or_gate).pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="Porte AND", command=self.load_and_gate).pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="Porte NOR", command=self.load_nor_gate).pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="Porte NAND", command=self.load_nand_gate).pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="Données personnalisées", command=self.load_custom_data).pack(fill=tk.X, pady=2)
        
        # --- Paramètres ---
        params_frame = ttk.LabelFrame(left_frame, text="Paramètres", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Taux d'apprentissage
        ttk.Label(params_frame, text="Taux d'apprentissage:").pack(anchor=tk.W)
        self.learning_rate_var = tk.DoubleVar(value=0.1)
        learning_rate_scale = ttk.Scale(params_frame, from_=0.01, to=1.0, 
                                       variable=self.learning_rate_var, orient=tk.HORIZONTAL)
        learning_rate_scale.pack(fill=tk.X)
        self.lr_label = ttk.Label(params_frame, text="0.1")
        self.lr_label.pack()
        learning_rate_scale.configure(command=lambda v: self.lr_label.configure(text=f"{float(v):.2f}"))
        
        # Nombre d'époques
        ttk.Label(params_frame, text="Époques par cycle:").pack(anchor=tk.W, pady=(10, 0))
        self.epochs_var = tk.IntVar(value=100)
        epochs_spinbox = ttk.Spinbox(params_frame, from_=10, to=10000, textvariable=self.epochs_var)
        epochs_spinbox.pack(fill=tk.X)
        
        # --- Contrôles d'entraînement ---
        training_frame = ttk.LabelFrame(left_frame, text="Entraînement", padding=10)
        training_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.train_button = ttk.Button(training_frame, text="▶ Démarrer l'entraînement", 
                                       command=self.start_training)
        self.train_button.pack(fill=tk.X, pady=2)
        
        self.step_button = ttk.Button(training_frame, text="⏭ Une époque", 
                                      command=self.train_one_epoch)
        self.step_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(training_frame, text="⏸ Arrêter", 
                                      command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        self.reset_button = ttk.Button(training_frame, text="🔄 Réinitialiser", 
                                       command=self.reset_neurone)
        self.reset_button.pack(fill=tk.X, pady=2)
        
        # --- Informations ---
        info_frame = ttk.LabelFrame(left_frame, text="Informations", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.info_text = tk.Text(info_frame, height=15, width=35, bg='#1e1e1e', 
                                fg='#00ff00', font=('Courier', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # ===== PARTIE DROITE: Visualisations =====
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Notebook pour les onglets
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Onglet 1: Structure du neurone
        self.structure_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.structure_tab, text="🔬 Structure du Neurone")
        
        # Onglet 2: Courbes d'apprentissage
        self.learning_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.learning_tab, text="📊 Apprentissage")
        
        # Onglet 3: Prédictions
        self.predictions_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.predictions_tab, text="🎯 Prédictions")
        
        # Créer les graphiques
        self.create_structure_plot()
        self.create_learning_plots()
        self.create_predictions_plot()
        
    def create_structure_plot(self):
        """Créer la visualisation de la structure du neurone"""
        self.fig_structure = Figure(figsize=(8, 6), facecolor='#2b2b2b')
        self.ax_structure = self.fig_structure.add_subplot(111)
        self.ax_structure.set_facecolor('#1e1e1e')
        
        self.canvas_structure = FigureCanvasTkAgg(self.fig_structure, self.structure_tab)
        self.canvas_structure.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_learning_plots(self):
        """Créer les graphiques d'apprentissage"""
        self.fig_learning = Figure(figsize=(8, 6), facecolor='#2b2b2b')
        
        # Sous-graphiques
        self.ax_error = self.fig_learning.add_subplot(221)
        self.ax_weights = self.fig_learning.add_subplot(222)
        self.ax_bias = self.fig_learning.add_subplot(223)
        self.ax_error_log = self.fig_learning.add_subplot(224)
        
        for ax in [self.ax_error, self.ax_weights, self.ax_bias, self.ax_error_log]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
        
        self.fig_learning.tight_layout()
        
        self.canvas_learning = FigureCanvasTkAgg(self.fig_learning, self.learning_tab)
        self.canvas_learning.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_predictions_plot(self):
        """Créer la visualisation des prédictions"""
        self.fig_predictions = Figure(figsize=(8, 6), facecolor='#2b2b2b')
        self.ax_predictions = self.fig_predictions.add_subplot(111)
        self.ax_predictions.set_facecolor('#1e1e1e')
        
        self.canvas_predictions = FigureCanvasTkAgg(self.fig_predictions, self.predictions_tab)
        self.canvas_predictions.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def draw_neuron_structure(self):
        """Dessiner la structure du neurone avec ses connexions"""
        if self.neurone is None:
            return
            
        self.ax_structure.clear()
        self.ax_structure.set_xlim(-1, 4)
        self.ax_structure.set_ylim(-1, 4)
        self.ax_structure.axis('off')
        self.ax_structure.set_facecolor('#1e1e1e')
        
        n_inputs = len(self.neurone.weights)
        
        # Position des neurones d'entrée
        input_y_positions = np.linspace(0.5, 3.5, n_inputs)
        input_x = 0.5
        
        # Position du neurone de sortie
        output_x = 3
        output_y = 2
        
        # Dessiner les connexions avec les poids
        for i, (y_pos, weight) in enumerate(zip(input_y_positions, self.neurone.weights)):
            # Épaisseur proportionnelle au poids
            linewidth = abs(weight) * 3 + 0.5
            color = 'green' if weight > 0 else 'red'
            alpha = min(abs(weight), 1.0)
            
            self.ax_structure.plot([input_x + 0.15, output_x - 0.15], 
                                 [y_pos, output_y],
                                 color=color, linewidth=linewidth, alpha=alpha)
            
            # Afficher la valeur du poids
            mid_x = (input_x + output_x) / 2
            mid_y = (y_pos + output_y) / 2
            self.ax_structure.text(mid_x, mid_y, f'{weight:.2f}', 
                                 fontsize=9, color='yellow',
                                 bbox=dict(boxstyle='round', facecolor='#2b2b2b', alpha=0.8))
        
        # Dessiner les neurones d'entrée
        for i, y_pos in enumerate(input_y_positions):
            circle = patches.Circle((input_x, y_pos), 0.15, 
                                   facecolor='#4CAF50', edgecolor='white', linewidth=2)
            self.ax_structure.add_patch(circle)
            self.ax_structure.text(input_x - 0.4, y_pos, f'x{i+1}', 
                                 fontsize=12, color='white', va='center')
        
        # Dessiner le neurone de sortie
        circle = patches.Circle((output_x, output_y), 0.2, 
                               facecolor='#2196F3', edgecolor='white', linewidth=3)
        self.ax_structure.add_patch(circle)
        self.ax_structure.text(output_x, output_y, 'Σ', 
                             fontsize=16, color='white', ha='center', va='center')
        self.ax_structure.text(output_x + 0.4, output_y, 'Output', 
                             fontsize=12, color='white', va='center')
        
        # Afficher le biais
        bias_y = -0.3
        self.ax_structure.text(output_x, bias_y, f'Biais: {self.neurone.bias:.3f}', 
                             fontsize=11, color='cyan',
                             bbox=dict(boxstyle='round', facecolor='#2b2b2b', 
                                     edgecolor='cyan', linewidth=2))
        
        # Titre
        self.ax_structure.text(2, 3.8, 'Structure du Neurone', 
                             fontsize=14, color='white', ha='center', weight='bold')
        
        self.canvas_structure.draw()
        
    def update_learning_plots(self):
        """Mettre à jour les graphiques d'apprentissage"""
        if not self.errors_history:
            return
            
        # Erreur
        self.ax_error.clear()
        self.ax_error.plot(self.errors_history, color='#FF5252', linewidth=2)
        self.ax_error.set_title('Erreur moyenne', color='white', fontsize=10)
        self.ax_error.set_xlabel('Époque', color='white')
        self.ax_error.set_ylabel('Erreur', color='white')
        self.ax_error.grid(True, alpha=0.2, color='white')
        self.ax_error.set_facecolor('#1e1e1e')
        
        # Poids
        self.ax_weights.clear()
        weights_array = np.array(self.weights_history)
        for i in range(weights_array.shape[1]):
            self.ax_weights.plot(weights_array[:, i], label=f'w{i+1}', linewidth=2)
        self.ax_weights.set_title('Évolution des poids', color='white', fontsize=10)
        self.ax_weights.set_xlabel('Époque', color='white')
        self.ax_weights.set_ylabel('Valeur', color='white')
        self.ax_weights.legend(loc='upper right')
        self.ax_weights.grid(True, alpha=0.2, color='white')
        self.ax_weights.set_facecolor('#1e1e1e')
        
        # Biais
        self.ax_bias.clear()
        self.ax_bias.plot(self.bias_history, color='#4CAF50', linewidth=2)
        self.ax_bias.set_title('Évolution du biais', color='white', fontsize=10)
        self.ax_bias.set_xlabel('Époque', color='white')
        self.ax_bias.set_ylabel('Valeur', color='white')
        self.ax_bias.grid(True, alpha=0.2, color='white')
        self.ax_bias.set_facecolor('#1e1e1e')
        
        # Erreur en log
        self.ax_error_log.clear()
        self.ax_error_log.plot(self.errors_history, color='#9C27B0', linewidth=2)
        self.ax_error_log.set_yscale('log')
        self.ax_error_log.set_title('Erreur (échelle log)', color='white', fontsize=10)
        self.ax_error_log.set_xlabel('Époque', color='white')
        self.ax_error_log.set_ylabel('Erreur (log)', color='white')
        self.ax_error_log.grid(True, alpha=0.2, color='white')
        self.ax_error_log.set_facecolor('#1e1e1e')
        
        self.fig_learning.tight_layout()
        self.canvas_learning.draw()
        
    def update_predictions_plot(self):
        """Mettre à jour la visualisation des prédictions"""
        if self.neurone is None or self.training_data is None:
            return
            
        self.ax_predictions.clear()
        
        X = self.training_data['X']
        y = self.training_data['y']
        
        # Prédictions
        predictions = []
        for x in X:
            pred = self.neurone.predict(x)
            predictions.append(pred)
        
        # Afficher les points
        for i, (x_val, target, pred) in enumerate(zip(X, y, predictions)):
            color = '#4CAF50' if abs(pred - target) < 0.1 else '#FF5252'
            marker = 'o' if target == 1 else 's'
            self.ax_predictions.scatter(i, pred, c=color, s=200, marker=marker, 
                                      edgecolors='white', linewidths=2, alpha=0.7)
            self.ax_predictions.text(i, pred + 0.1, f'{pred:.3f}', 
                                   ha='center', color='white', fontsize=9)
            
        # Ligne de référence
        self.ax_predictions.axhline(y=0.5, color='yellow', linestyle='--', 
                                  linewidth=2, alpha=0.5, label='Seuil (0.5)')
        
        self.ax_predictions.set_ylim(-0.1, 1.1)
        self.ax_predictions.set_xlabel('Exemple', color='white')
        self.ax_predictions.set_ylabel('Prédiction', color='white')
        self.ax_predictions.set_title('Prédictions vs Cibles', color='white', fontsize=12, weight='bold')
        self.ax_predictions.legend()
        self.ax_predictions.grid(True, alpha=0.2, color='white')
        self.ax_predictions.set_facecolor('#1e1e1e')
        
        # Définir les labels de l'axe x
        labels = [f'{x[0]},{x[1]}' for x in X]
        self.ax_predictions.set_xticks(range(len(X)))
        self.ax_predictions.set_xticklabels(labels, color='white')
        
        self.canvas_predictions.draw()
        
    def update_info_text(self):
        """Mettre à jour le texte d'information"""
        if self.neurone is None:
            return
            
        self.info_text.delete('1.0', tk.END)
        
        info = f"""
╔══════════════════════════════════╗
║   INFORMATIONS DU NEURONE        ║
╚══════════════════════════════════╝

Époque actuelle: {self.current_epoch}

--- Paramètres ---
Poids: {[f'{w:.4f}' for w in self.neurone.weights]}
Biais: {self.neurone.bias:.4f}

--- Performance ---
"""
        
        if self.errors_history:
            info += f"Erreur actuelle: {self.errors_history[-1]:.6f}\n"
            info += f"Erreur initiale: {self.errors_history[0]:.6f}\n"
            reduction = (1 - self.errors_history[-1] / self.errors_history[0]) * 100
            info += f"Réduction: {reduction:.2f}%\n"
        
        if self.training_data is not None:
            info += "\n--- Prédictions ---\n"
            X = self.training_data['X']
            y = self.training_data['y']
            
            correct = 0
            for x, target in zip(X, y):
                pred = self.neurone.predict(x)
                pred_binary = 1 if pred >= 0.5 else 0
                is_correct = "✓" if pred_binary == target else "✗"
                if pred_binary == target:
                    correct += 1
                info += f"{is_correct} [{x[0]},{x[1]}] → {pred:.3f} "
                info += f"(cible: {target})\n"
            
            accuracy = (correct / len(X)) * 100
            info += f"\nPrécision: {accuracy:.1f}%\n"
        
        self.info_text.insert('1.0', info)
        
    def load_or_gate(self):
        """Charger les données de la porte OR"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        self.training_data = {'X': X, 'y': y, 'name': 'OR'}
        self.reset_neurone()
        
    def load_and_gate(self):
        """Charger les données de la porte AND"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        self.training_data = {'X': X, 'y': y, 'name': 'AND'}
        self.reset_neurone()
        
    def load_nor_gate(self):
        """Charger les données de la porte NOR"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([1, 0, 0, 0])
        self.training_data = {'X': X, 'y': y, 'name': 'NOR'}
        self.reset_neurone()
        
    def load_nand_gate(self):
        """Charger les données de la porte NAND"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([1, 1, 1, 0])
        self.training_data = {'X': X, 'y': y, 'name': 'NAND'}
        self.reset_neurone()
        
    def load_custom_data(self):
        """Charger des données personnalisées (à implémenter)"""
        messagebox.showinfo("Info", "Fonctionnalité à venir!")
        
    def reset_neurone(self):
        """Réinitialiser le neurone"""
        if self.training_data is None:
            return
            
        n_inputs = self.training_data['X'].shape[1]
        self.neurone = Neurone(n_inputs)
        self.current_epoch = 0
        self.errors_history = []
        self.weights_history = []
        self.bias_history = []
        
        self.update_all_visualizations()
        
    def train_one_epoch(self):
        """Entraîner le neurone pour une époque"""
        if self.neurone is None or self.training_data is None:
            return
            
        X = self.training_data['X']
        y = self.training_data['y']
        lr = self.learning_rate_var.get()
        
        epoch_errors = []
        for i in range(len(X)):
            error = self.neurone.train(X[i], y[i], learning_rate=lr)
            epoch_errors.append(error)
        
        self.current_epoch += 1
        avg_error = np.mean(epoch_errors)
        self.errors_history.append(avg_error)
        self.weights_history.append(self.neurone.weights.copy())
        self.bias_history.append(self.neurone.bias)
        
        self.update_all_visualizations()
        
    def start_training(self):
        """Démarrer l'entraînement continu"""
        if self.neurone is None or self.training_data is None:
            return
            
        self.is_training = True
        self.train_button.config(state=tk.DISABLED)
        self.step_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        self.train_loop()
        
    def train_loop(self):
        """Boucle d'entraînement"""
        if not self.is_training:
            return
            
        epochs_per_cycle = self.epochs_var.get()
        
        for _ in range(min(10, epochs_per_cycle)):  # 10 époques à la fois pour la réactivité
            self.train_one_epoch()
            
        if self.is_training:
            self.root.after(50, self.train_loop)  # Continuer après 50ms
            
    def stop_training(self):
        """Arrêter l'entraînement"""
        self.is_training = False
        self.train_button.config(state=tk.NORMAL)
        self.step_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
    def update_all_visualizations(self):
        """Mettre à jour toutes les visualisations"""
        self.draw_neuron_structure()
        self.update_learning_plots()
        self.update_predictions_plot()
        self.update_info_text()


def main():
    root = tk.Tk()
    app = NeuroneGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
