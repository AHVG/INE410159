# Detecção de Embaúba (Cecropia) — Visão Computacional Clássica

Projeto final da disciplina de Visão Computacional — UFSC.

Detecta árvores **Cecropia sp. (Embaúba)** em imagens aéreas usando segmentação por cor no espaço HSV, sem uso de deep learning.

## Pipeline

```
Tile JPEG → Gaussian Blur → Segmentação HSV → CLOSE → OPEN → Filtro de área → Convex Hull
```

| Parâmetro | Valor |
|-----------|-------|
| Espaço de cor | HSV |
| Faixa de detecção | H:[44,58] S:[144,231] V:[104,203] |
| Morfologia CLOSE | elipse 25×25 px |
| Morfologia OPEN | elipse 9×9 px |
| Área mínima | 10.000 px² |

## Dataset

- 992 tiles JPEG de 2048×2048 px com 25% de sobreposição
- Mosaico aéreo georeferenciado em EPSG:31982 (UTM Zone 22S)

## Instalação

```bash
pip install opencv-python matplotlib numpy
```

## Uso

```bash
MPLBACKEND=Agg python3 relatorio_embauba.py
```

Gera em `output/`:
- `passo_a_passo/tile_XXXX.png` — visualização das 8 etapas do pipeline para cada tile
- Histogramas: detecções, áreas, tempo, cobertura HSV, uso de memória
- `relatorio.md` — estatísticas completas

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `embaubaHSVmask.py` | Script base original (referência) |
| `relatorio_embauba.py` | Pipeline completo com processamento paralelo e relatório |
