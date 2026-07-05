# Relatório de Resultados — Contagem de Cecropia por Parcela
## Pipeline: SAM3 + YOLOv8x | MataDIV (ESALQ/USP)
**Autor:** Gabriel A. Ferreira Gualda
**Data:** 05/07/2026

---

## 1. Resumo Geral

| Métrica | Valor |
|---|---|
| Total de parcelas | 144 |
| Parcelas com Cecropia (plantadas > 0) | 60 |
| Total de CP plantadas | 2256 |
| Total de CP mapeadas | 1226 |
| Taxa de detecção global | 54.3% |
| Total com segmentação | 1228 |

## 2. Categorias de Detecção

| Categoria | Quantidade | Proporção |
|---|---|---|
| YOLO + SAM3 (ambos) | 564 | 46.0% |
| Só YOLO (novas) | 267 | 21.8% |
| Só SAM3 (YOLO não detectou) | 395 | 32.2% |
| **Total mapeadas** | **1226** | **100%** |

## 3. Detalhamento

| Fonte | Quantidade |
|---|---|
| Detecções YOLO (dentro parcelas) | 831 |
| Máscaras SAM3 text prompt | 961 |
| Máscaras SAM3 bbox prompt | 267 |
| SAM3 que YOLO não detectou | 395 |

## 4. Estatísticas de Área das Copas

| Estatística | Valor |
|---|---|
| N copas segmentadas | 1228 |
| Área média | 5.55 m² |
| Área mediana | 4.29 m² |
| Área mínima | 0.00 m² |
| Área máxima | 24.26 m² |
| Desvio padrão | 3.93 m² |

## 5. Resultados por Composição

| Comp | Parcelas | Plantadas | YOLO | só SAM3 | Total | Segm | Taxa (%) | Área média (m²) |
|---|---|---|---|---|---|---|---|---|
| D1 | 12 | 1200 | 300 | 141 | 441 | 442 | 36.8 | 3.9 |
| D10 | 12 | 0 | 8 | 2 | 10 | 10 | 0.0 | 5.7 |
| D11 | 12 | 204 | 83 | 45 | 128 | 129 | 62.7 | 8.1 |
| D12 | 12 | 60 | 34 | 7 | 41 | 41 | 68.3 | 5.8 |
| D2 | 13 | 0 | 20 | 2 | 22 | 22 | 0.0 | 4.3 |
| D3 | 12 | 0 | 15 | 3 | 18 | 19 | 0.0 | 4.1 |
| D4 | 12 | 0 | 15 | 2 | 17 | 16 | 0.0 | 5.4 |
| D5 | 11 | 0 | 10 | 1 | 11 | 11 | 0.0 | 5.0 |
| D6 | 12 | 0 | 6 | 7 | 13 | 14 | 0.0 | 6.7 |
| D7 | 12 | 396 | 174 | 87 | 261 | 260 | 65.9 | 6.1 |
| D8 | 12 | 396 | 163 | 97 | 260 | 259 | 65.7 | 7.0 |
| D9 | 12 | 0 | 3 | 1 | 4 | 5 | 0.0 | 6.3 |


## 6. Figuras Geradas

1. `fig01_plantadas_vs_detectadas.png` — Barras: plantadas vs detectadas por composição
2. `fig02_categorias_empilhadas.png` — Barras empilhadas: 3 categorias por composição
3. `fig03_histograma_areas.png` — Histograma: distribuição de áreas das copas
4. `fig04_boxplot_areas_composicao.png` — Boxplot: áreas por composição
5. `fig05_taxa_deteccao.png` — Barras: taxa de detecção (%) por composição
6. `fig06_donut_categorias.png` — Donut: proporção das 3 categorias

---

*Relatório gerado automaticamente por `contar_cp_por_parcela.py`*
