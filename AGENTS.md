# AGENTS.md — Guia para Agentes de IA

## Contexto do Projeto

Detecção de árvores **Cecropia (Embaúba)** em mosaico aéreo usando visão computacional clássica (sem deep learning). Projeto final da disciplina de Visão Computacional — UFSC.

## Estrutura de Arquivos

```
.
├── embaubaHSVmask.py   # Script base original — NÃO MODIFICAR
├── main.py             # Ponto de entrada — orquestra os módulos
├── deteccao.py         # Algoritmos de visão computacional (HSV, morfologia, hull)
├── execucao.py         # Processamento paralelo + gerenciador de checkpoint
├── estatisticas.py     # Figuras estatísticas + relatório Markdown
├── tiles/              # 992 tiles JPEG 2048×2048 px
│   └── metadados_tiles.json
└── output/             # Gerado pelo pipeline (ignorado no git)
    ├── passo_a_passo/      # 1 PNG por tile com 8 etapas do pipeline
    ├── checkpoints/        # Histórico de runs arquivadas
    ├── checkpoint_latest.json
    ├── hist_deteccoes_por_tile.png
    ├── hist_areas_deteccao.png
    ├── top10_tiles.png
    ├── tempo_processamento.png
    ├── cobertura_percentual.png
    ├── memoria_por_tile.png
    └── relatorio.md
```

## Regras Importantes

- **`embaubaHSVmask.py` é intocável** — serve apenas como referência do algoritmo base.
- Todo desenvolvimento acontece em `relatorio_embauba.py`, que é auto-contido (não importa de `embaubaHSVmask.py`).
- Não usar deep learning — apenas `cv2`, `numpy`, `matplotlib`.
- Não adicionar comentários óbvios; só quando o motivo não é evidente.

## Pipeline (em ordem)

| Etapa | Operação | Parâmetros |
|-------|----------|------------|
| 1 | Gaussian Blur | kernel 9×9 |
| 2 | Segmentação HSV | H:[44,58] S:[144,231] V:[104,203] |
| 3 | Morfologia CLOSE | elipse 25×25 (fecha buracos) |
| 4 | Morfologia OPEN | elipse 9×9 (remove ruído) |
| 5 | Filtro de contornos | área > 10.000 px² |
| 6 | Convex Hull | desenha o contorno da copa |

## Como Executar

```bash
MPLBACKEND=Agg python3 main.py
```

Requer: `opencv-python`, `matplotlib`, `numpy`.

## Dataset

- **992 tiles** JPEG, 2048×2048 px, sobreposição de 512 px (25%)
- CRS: EPSG:31982 (UTM Zone 22S, Brasil)
- Resolução: ~0.00848 u/px
- Tiles completamente pretos (fora da área mapeada) são ignorados automaticamente

## Saídas

- `output/passo_a_passo/tile_XXXX.png` — visualização das 8 etapas por tile
- Gráficos estatísticos em `output/`
- `output/relatorio.md` — relatório completo em Markdown
