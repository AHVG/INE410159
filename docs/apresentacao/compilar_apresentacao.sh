#!/bin/bash
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode apresentacao.tex
pdflatex -interaction=nonstopmode apresentacao.tex
rm -f apresentacao.aux apresentacao.log apresentacao.nav apresentacao.out apresentacao.snm apresentacao.toc
echo "PDF gerado: $(pwd)/apresentacao.pdf"
