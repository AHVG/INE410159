#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "Uso: ./dev.sh <comando> [args]"
    echo ""
    echo "  relatorio       Compila docs/relatorio/relatorio.tex → PDF"
    echo "  apresentacao    Compila docs/apresentacao/apresentacao.tex → PDF"
    echo "  pipeline        Roda o pipeline completo nos tiles → data/output/"
    echo "  figuras         Gera figuras qualitativas do relatório"
    echo "  anotar [alvo]   Abre a UI de validação (tiles/globs/pasta, --amostra N)"
    exit 1
}

check_python_deps() {
    python3 -c "import cv2, numpy, matplotlib" 2>/dev/null || {
        echo "[INFO] Instalando dependências Python..."
        pip3 install opencv-python matplotlib numpy --user --break-system-packages
    }
}

cmd="${1:-}"
shift || true

case "$cmd" in
    relatorio)
        cd "$ROOT/docs/relatorio"
        pdflatex -interaction=nonstopmode relatorio.tex
        pdflatex -interaction=nonstopmode relatorio.tex
        rm -f relatorio.aux relatorio.log relatorio.toc relatorio.out relatorio.lof relatorio.lot
        echo "[OK] PDF gerado: $ROOT/docs/relatorio/relatorio.pdf"
        ;;
    apresentacao)
        cd "$ROOT/docs/apresentacao"
        pdflatex -interaction=nonstopmode apresentacao.tex
        pdflatex -interaction=nonstopmode apresentacao.tex
        rm -f apresentacao.aux apresentacao.log apresentacao.nav apresentacao.out apresentacao.snm apresentacao.toc apresentacao.vrb
        echo "[OK] PDF gerado: $ROOT/docs/apresentacao/apresentacao.pdf"
        ;;
    pipeline)
        check_python_deps
        if [ ! -f "$ROOT/data/tiles/metadados_tiles.json" ]; then
            echo "[ERRO] Tiles não encontrados em data/tiles/"
            echo "       Download: https://drive.google.com/drive/folders/1Gux57VofI_bxgt66hlXgaVlp_s21emNl"
            exit 1
        fi
        mkdir -p "$ROOT/data/output/tiles" "$ROOT/data/output/checkpoints"
        MPLBACKEND=Agg python3 "$ROOT/src/main.py"
        echo "[OK] Resultados em data/output/"
        ;;
    figuras)
        check_python_deps
        MPLBACKEND=Agg python3 "$ROOT/src/main.py" --etapas tile_0681
        MPLBACKEND=Agg python3 "$ROOT/src/main.py" --exemplos
        ;;
    anotar)
        check_python_deps
        python3 "$ROOT/src/anotar.py" "$@"
        ;;
    *)
        usage
        ;;
esac
