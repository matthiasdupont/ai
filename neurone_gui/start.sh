#!/bin/bash

# Script de lancement rapide pour macOS
# Usage: ./start.sh

echo "🧠 Lancement de l'interface graphique du neurone..."
echo ""

# Se placer dans le bon répertoire
cd "$(dirname "$0")"

# Activer l'environnement virtuel si disponible
if [ -d "../.venv" ]; then
    echo "📦 Activation de l'environnement virtuel..."
    source ../.venv/bin/activate
fi

# Vérifier Python
if ! command -v python &> /dev/null; then
    echo "❌ Python n'est pas installé!"
    exit 1
fi

# Afficher la version
echo "🐍 Python $(python --version)"
echo ""

# Lancer l'interface
echo "▶️  Démarrage de l'interface..."
echo ""
python neurone_gui.py

# Si erreur
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Erreur lors du lancement!"
    echo "💡 Essayez:"
    echo "   python test_gui.py"
    exit 1
fi
