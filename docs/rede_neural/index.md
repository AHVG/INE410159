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
| `relatorio.md` | Transcrição editável do texto do PDF (Introdução → Métodos). |
| `figuras/` | Imagens referenciadas no texto (`[FIGURA X]` do original). A preencher. |
| `resultados/` | Métricas e figuras de resultados. A preencher (não existem ainda). |
| `index.md` | Este arquivo. |

## Estado do material

O PDF original **termina em "3.3.7 Métricas de avaliação"**. Está completo até a
metodologia; **faltam Resultados, Discussão e Conclusão**, e todas as figuras são
placeholders `[FIGURA X]`.

No relatório final ([`../relatorio/relatorio.tex`](../relatorio/relatorio.tex)),
esses vãos aparecem como marcadores `\pendente{...}` (em vermelho). Ao preencher
esta pasta, atualizar os marcadores correspondentes no `.tex` e no `.md`.

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

### Resultados → `resultados/`
- [ ] Métricas de detecção no teste: Precisão, Recall, F1, mAP50, mAP50-95.
- [ ] Curva Precision-Recall e curvas de treino (loss / mAP por época).
- [ ] Importância dos hiperparâmetros (Random Forest / Mean Decrease Impurity).
- [ ] Contagem final de indivíduos mapeados por categoria (i / ii / iii).
- [ ] Figuras qualitativas de detecção e segmentação sobre o ortomosaico.
- [ ] Texto de Discussão e Conclusão da frente neural.

## Onde isso entra no relatório final

A estrutura combinada está descrita em
[`../relatorio/index.md`](../relatorio/index.md). A frente neural corresponde à
**Parte II** do relatório e aos slides da **Parte II** da apresentação
([`../apresentacao/index.md`](../apresentacao/index.md)).
