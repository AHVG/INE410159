# AGENTS.md — Guia para Agentes de IA

## Contexto do Projeto

Detecção de árvores **Cecropia (Embaúba)** em mosaico aéreo usando visão computacional clássica (sem deep learning). Projeto final da disciplina de Visão Computacional — UFSC.

Este repositório corresponde à parte de **visão computacional clássica** de um
trabalho maior sobre detecção e segmentação de copas de **Cecropia
pachystachya** em ortomosaicos de VANT. A outra frente do trabalho usa **SAM 3 e
YOLOv8** para detecção/segmentação. Neste repositório, manter a restrição de não
usar deep learning: apenas `cv2`, `numpy` e `matplotlib`.

## Grupo 10

- Gabriel Arantes F. Gualda
- Fábio H. A. Coelho
- Nena A. Impanta
- Augusto de H. V. Guerner
- Eduardo G. de Lazari

## Estrutura de Arquivos

```
.
├── README.md  AGENTS.md  CLAUDE.md  index.md   # documentação
├── dev.sh                                       # compila docs ou roda o projeto
├── .gitignore
├── src/                    # CÓDIGO
│   ├── core/               #   Núcleo (lógica/helpers), importado pelos pontos de entrada
│   │   ├── deteccao.py     #     VC clássica + detectar_embaubas(img) → bounds (saída canônica)
│   │   ├── execucao.py     #     Processamento paralelo + gerenciador de checkpoint
│   │   └── estatisticas.py #     Figuras estatísticas + relatório Markdown
│   ├── ref/                #   Referência — NÃO usar como código do projeto
│   │   └── embaubaHSVmask.py #   Script base original do professor — NÃO MODIFICAR/IMPORTAR
│   ├── main.py             #   Entrada principal: output completo ou JSON por path
│   └── anotar.py           #   UI OpenCV para revisar labels e marcar faltantes
├── docs/                   # DOCUMENTAÇÃO
│   ├── relatorio/          #   Relatório escrito (relatorio.tex, relatorio.pdf)
│   └── apresentacao/       #   Slides (apresentacao.tex, apresentacao.pdf)
└── data/                   # DADOS e artefatos
    ├── tiles/              #   992 tiles JPEG 2048×2048 px (+ metadados_tiles.json)
    ├── validacao/          #   Conjunto rotulado único (ver data/validacao/index.md)
    └── output/             #   Gerado pelo pipeline (ignorado no git)
```

Os pontos de entrada em `src/` importam de `core.*`; os caminhos de dados são
resolvidos a partir de `__file__`, então rodam de qualquer diretório.
O conteúdo de `data/output/` (passo a passo por tile, histogramas, `relatorio.md`)
é gerado pelo `src/main.py`.

## Estrutura do Relatório

O arquivo [`docs/relatorio/index.md`](docs/relatorio/index.md) define a estrutura obrigatória do `data/output/relatorio.md`. Ao gerar ou atualizar o relatório, seguir exatamente as seções e requisitos descritos lá: Introdução, Dataset, Pipeline (com motivação e parâmetros de cada etapa), Resultados e Conclusão.

## Regras Importantes

- **`src/ref/embaubaHSVmask.py` é intocável** — serve apenas como referência do algoritmo base.
- Todo desenvolvimento acontece em `src/`, sem importar `src/ref/embaubaHSVmask.py`.
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
| 6 | Filtro `eh_embauba` | área ≥ 33.798 ou circularidade ≤ 0.1551 |
| 7 | Convex Hull | desenha o contorno da copa |

## Como Executar

As entradas públicas são `src/main.py` e `src/anotar.py`:

```bash
./dev.sh pipeline                            # pipeline completo → data/output/
./dev.sh anotar tile_0905                    # revisa labels e faltantes da validação
./dev.sh anotar data/validacao --pendentes
./dev.sh relatorio                           # compila docs/relatorio/relatorio.pdf
./dev.sh apresentacao                        # compila docs/apresentacao/apresentacao.pdf

# Direto:
MPLBACKEND=Agg python3 src/main.py           # pipeline completo nos tiles → data/output/
python3 src/main.py <imagem_ou_pasta>         # detecção avulsa → JSON por imagem
python3 src/main.py <imagem_ou_pasta> --vis --passo-a-passo
```

Requer: `opencv-python`, `matplotlib`, `numpy`.

## Dataset

- **992 tiles** JPEG, 2048×2048 px
- CRS: EPSG:31982 (UTM Zone 22S, Brasil)
- Resolução: ~0.00848 u/px
- Tiles completamente pretos (fora da área mapeada) são ignorados automaticamente

O conjunto rotulado único de validação e o esquema dos JSONs de anotação estão
documentados em [`data/validacao/index.md`](data/validacao/index.md).

## Saídas

- `data/output/passo_a_passo/tile_XXXX.png` — visualização das 8 etapas por tile
- Gráficos estatísticos em `data/output/`
- `data/output/relatorio.md` — relatório completo em Markdown
