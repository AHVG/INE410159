#!/bin/bash
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode relatorio.tex
pdflatex -interaction=nonstopmode relatorio.tex
rm -f relatorio.aux relatorio.log relatorio.toc relatorio.out relatorio.lof relatorio.lot
echo "PDF gerado: $(pwd)/relatorio.pdf"
