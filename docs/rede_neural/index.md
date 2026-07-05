# Material-fonte — Solução com Redes Neurais (SAM 3 + YOLOv8)

Esta pasta guarda o **material-fonte da frente de redes neurais** do trabalho, que
serve de **entrada** para a geração do relatório e da apresentação finais
combinados. Ela **não** é compilada: é insumo editável e versionado.

O relatório final que integra as duas soluções fica em
[`../relatorio/`](../relatorio/); a apresentação em [`../apresentacao/`](../apresentacao/).

## Arquivos desta pasta

| Arquivo | Descrição |
|---------|-----------|
| `sam3_yolo_detection.pdf` | PDF original da frente neural (fonte imutável). |
| `relatorio.md` | Transcrição editável do texto do PDF e síntese dos resultados disponíveis. |
| `figuras/` | Imagens referenciadas no texto e figuras quantitativas geradas. |
| `resultados/` | CSVs, relatório automático e métricas consolidadas por parcela/composição. |
| `index.md` | Este arquivo. |

## Estado do material

O PDF original **termina em "3.3.7 Métricas de avaliação"**. A nova leva de
resultados acrescenta a contagem final por parcela/composição, estatísticas de
área das copas segmentadas e seis figuras quantitativas. Ainda não foram incluídas
as métricas de teste do detector (Precisão, Recall, F1, mAP50, mAP50-95), curvas
de treino/PR, importância de hiperparâmetros e figuras qualitativas.

## Como preencher (checklist)

### Figuras do texto → `figuras/`
- [ ] `mapa_localizacao.png` — Brasil / SP / Itatinga / MataDIV (§3.1)
- [ ] `tabela1_composicoes.*` — composições de espécies do MataDIV (§3.1)
- [ ] `fluxograma_pipeline.png` — as 6 etapas do pipeline (§3.3)
- [ ] `exemplos_copas.png` — copa isolada / adjacente / jovem (§3.3)
- [ ] `correcao_anotacoes.png` — verde (anotações) × vermelho (detecções) (§3.3.3)
- [ ] `rodadas_inferencia.png` — grade regular × deslocada (§3.3.5)
- [ ] `cruzamento_categorias.png` — YOLO × SAM 3, 3 categorias (§3.3.5)
- [ ] `segmentacao_bbox_prompt.png` — recorte / máscara / polígono (§3.3.6)
- [ ] `iou_exemplo.png` — cálculo de IoU (§3.3.7)
- [x] `fig01_plantadas_vs_detectadas.png` — plantadas vs. detectadas por composição.
- [x] `fig02_categorias_empilhadas.png` — categorias YOLO/SAM3 por composição.
- [x] `fig03_histograma_areas.png` — distribuição das áreas segmentadas.
- [x] `fig04_boxplot_areas_composicao.png` — áreas por composição.
- [x] `fig05_taxa_deteccao.png` — taxa de detecção por composição.
- [x] `fig06_donut_categorias.png` — proporção das categorias de mapeamento.

### Resultados → `resultados/`
- [ ] Métricas de detecção no teste: Precisão, Recall, F1, mAP50, mAP50-95.
- [ ] Curva Precision-Recall e curvas de treino (loss / mAP por época).
- [ ] Importância dos hiperparâmetros (Random Forest / Mean Decrease Impurity).
- [x] Contagem final de indivíduos mapeados por categoria (i / ii / iii).
- [x] Contagem por parcela e resumo por composição.
- [x] Estatísticas de área das copas segmentadas.
- [ ] Figuras qualitativas de detecção e segmentação sobre o ortomosaico.
- [x] Texto preliminar de Resultados, Discussão e Conclusão com base nas contagens.

## Onde isso entra no relatório final

A estrutura combinada está descrita em
[`../relatorio/index.md`](../relatorio/index.md). A frente neural corresponde à
**Parte II** do relatório e aos slides da **Parte II** da apresentação
([`../apresentacao/index.md`](../apresentacao/index.md)).
