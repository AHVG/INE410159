# Material-fonte — Solução com Redes Neurais (SAM 3 + YOLOv8)

Esta pasta guarda o **material-fonte da frente de redes neurais** do trabalho, que
serve de **entrada** para a geração do relatório e da apresentação finais
combinados. Ela **não** é compilada: é insumo editável e versionado.

O relatório final que integra as duas soluções fica em
[`../relatorio/`](../relatorio/); a apresentação em [`../apresentacao/`](../apresentacao/).

## Arquivos desta pasta

| Arquivo | Descrição |
|---------|-----------|
| `codigo/` | Scripts Python do pipeline SAM 3 + YOLOv8 (ver seção [Código](#código-codigo)). |
| `sam3_yolo_detection.pdf` | PDF original da frente neural (fonte imutável). |
| `relatorio.md` | Transcrição editável do texto do PDF e síntese dos resultados disponíveis. |
| `figuras/` | Imagens referenciadas no texto e figuras quantitativas geradas. |
| `resultados/` | CSVs, relatório automático e métricas consolidadas por parcela/composição. |
| `index.md` | Este arquivo. |

## Código (`codigo/`)

Scripts Python que implementam o pipeline semi-automatizado **SAM 3 + YOLOv8** que
produziu os resultados desta pasta. São numerados na ordem de execução e usam
`transformers` (SAM 3), `ultralytics` (YOLOv8), `optuna`, `scikit-learn` e a pilha
geoespacial (`rasterio`, `geopandas`, `shapely`). **Não rodam neste repositório**:
dependem do ortomosaico georreferenciado, de GPU e dos pesos dos modelos — servem
como registro reprodutível do método descrito no `sam3_yolo_detection.pdf`.

O pipeline segue seis etapas (os scripts abaixo cobrem o método do PDF; a divisão
espacial em si — clusterização por grafo/BFS + Monte Carlo — gera os shapefiles de
*split* consumidos por `02`, mas não tem script dedicado nesta pasta):

**1. Geração e preparação do dataset (SAM 3 + curadoria)**

| Script | Função |
|--------|--------|
| `00_corte_orto_mask_SAM3.py` | Recorta o ortomosaico em tiles e roda o SAM 3 por *text prompt* (`"tree"`), exportando as copas candidatas como *shapefile*. |
| `01_split_labels.py` | Cria os vetores (*bounding boxes*) de recorte dos tiles a partir dos polígonos rotulados e filtra geometrias inválidas. |
| `02_create_dataset_model.py` | Recorta as imagens e os *labels* dos tiles para os conjuntos train/val/test a partir dos *shapefiles* de *split* (rasterio/geopandas). |

**3. Correção semi-automatizada das anotações**

| Script | Função |
|--------|--------|
| `08_encontrar_nao_anotadas_val.py` | Aplica o modelo na validação e sinaliza detecções sem anotação (IoU < 0,5) para revisão. |
| `09_encontrar_nao_anotadas_test.py` | Idem para o conjunto de teste. |
| `10_merge_labels_val.py` / `10_merge_labels_test.py` | Incorporam as anotações confirmadas aos labels (com backup e limpeza de cache). |

**4. Treinamento do YOLOv8x**

| Script | Função |
|--------|--------|
| `03_bayesian_optimization.py` | Otimização bayesiana de hiperparâmetros (Optuna/TPESampler, 30 tentativas). |
| `04_bayesian_analysis.py` | Analisa as tentativas e estima a importância dos hiperparâmetros (Random Forest / MDI). |
| `05_training_and_testing_model_AdamW_v2.py` | Treino final do YOLOv8x (AdamW) com os labels corrigidos. |

**5. Inferência e pós-processamento**

| Script | Função |
|--------|--------|
| `06_inference_and_postprocess_final_v2.py` | Inferência do YOLOv8x em duas rodadas (grade + *offset* 50%), georreferência, deduplicação e cruzamento com as máscaras SAM 3 (3 categorias). |
| `11_pos_processamento_inferencia.py` | Filtra as detecções por confiança e área (remove *bbox* grande com baixa confiança e *bbox* muito pequeno) e exporta GPKG/SHP limpos. |

**6. Segmentação das copas por *bbox prompt* (SAM 3)**

| Script | Função |
|--------|--------|
| `12_recortar_bbox_ortomosaico.py` | Recorta do ortomosaico a região de cada *bounding box* detectada (com *padding*). |
| `15a_sam3_bbox_prompt.py` | Segmenta as copas com SAM 3 usando a *bbox* como *prompt* geométrico. |
| `15b_sam2_bbox_prompt.py` | Variante com SAM 2 para comparação. |
| `15_comparacao_sam2_sam3_bbox.py` | Compara a segmentação por *bbox prompt* entre SAM 2 e SAM 3. |
| `16_teste_sistematico_sam3_bbox_cropsize.py` | Varredura sistemática do tamanho de recorte (*crop size*) no SAM 3 por *bbox*. |

**Consolidação dos resultados**

| Script | Função |
|--------|--------|
| `contagem_completa.py` | Contagem final de indivíduos por parcela/composição, base dos CSVs e figuras em `resultados/` e `figuras/`. |

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
