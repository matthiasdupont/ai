"""
Mode démo automatique pour l'interface graphique du neurone
Lance une démo automatique qui teste toutes les portes logiques
"""

import tkinter as tk
from tkinter import ttk
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neurone_gui.neurone_gui import NeuroneGUI


class NeuroneGUIDemo(NeuroneGUI):
    """Extension de l'interface avec mode démo"""
    
    def __init__(self, root):
        super().__init__(root)
        self.demo_mode = False
        self.demo_step = 0
        self.gates = ['OR', 'AND', 'NOR', 'NAND']
        self.current_gate_index = 0
        
        # Ajouter bouton démo
        demo_button = ttk.Button(self.root, text="🎬 Lancer la Démo", 
                                command=self.start_demo)
        demo_button.pack(side=tk.BOTTOM, pady=5)
        
    def start_demo(self):
        """Démarrer la démo automatique"""
        self.demo_mode = True
        self.demo_step = 0
        self.current_gate_index = 0
        self.run_demo()
        
    def run_demo(self):
        """Exécuter la démo étape par étape"""
        if not self.demo_mode:
            return
            
        if self.demo_step == 0:
            # Charger la première porte
            gate = self.gates[self.current_gate_index]
            if gate == 'OR':
                self.load_or_gate()
            elif gate == 'AND':
                self.load_and_gate()
            elif gate == 'NOR':
                self.load_nor_gate()
            elif gate == 'NAND':
                self.load_nand_gate()
            
            self.info_text.insert('1.0', f"\n\n🎬 DÉMO: Apprentissage porte {gate}\n")
            self.demo_step = 1
            self.root.after(2000, self.run_demo)
            
        elif self.demo_step == 1:
            # Entraîner
            self.epochs_var.set(50)
            self.learning_rate_var.set(0.5)
            self.start_training()
            self.demo_step = 2
            self.root.after(5000, self.run_demo)
            
        elif self.demo_step == 2:
            # Arrêter et passer à la porte suivante
            self.stop_training()
            self.current_gate_index += 1
            
            if self.current_gate_index < len(self.gates):
                self.demo_step = 0
                self.root.after(3000, self.run_demo)
            else:
                self.demo_mode = False
                self.info_text.insert('1.0', "\n\n✅ DÉMO TERMINÉE!\n")


def main():
    root = tk.Tk()
    app = NeuroneGUIDemo(root)
    root.mainloop()


if __name__ == '__main__':
    main()
