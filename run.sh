#!/bin/bash
set -e

echo "=== Detecção de Embaúba (Cecropia) ==="

# Verifica dependências Python
python3 -c "import cv2, numpy, matplotlib" 2>/dev/null || {
    echo "[INFO] Instalando dependências..."
    pip3 install opencv-python matplotlib numpy --user --break-system-packages
}

# Verifica tiles
if [ ! -f "tiles/metadados_tiles.json" ]; then
    echo "[ERRO] Pasta tiles/ não encontrada."
    echo "       Baixe os tiles em: https://drive.google.com/drive/folders/1Gux57VofI_bxgt66hlXgaVlp_s21emNl"
    exit 1
fi

# Cria pastas de saída
mkdir -p output/passo_a_passo output/checkpoints

# Executa pipeline
MPLBACKEND=Agg python3 main.py

echo ""
echo "[CONCLUÍDO] Resultados em output/"
