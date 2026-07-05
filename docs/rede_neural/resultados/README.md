# resultados/

Métricas e tabelas consolidadas da frente neural (SAM 3 + YOLOv8). Estes arquivos
complementam o PDF original, que terminava em "3.3.7 Métricas de avaliação".

## Arquivos

| Arquivo | Conteúdo |
|---------|----------|
| `relatorio_resultados.md` | Síntese automática das contagens, categorias e áreas. |
| `parcelas_contagem_completa.csv` | Contagem por parcela, composição, categoria e estatísticas de área. |
| `resumo_por_composicao.csv` | Agregados por composição/estrato. |
| `areas_copas_m2.csv` | Área individual das 1.228 copas segmentadas. |

## Principais resultados

- 144 parcelas avaliadas, 60 com _Cecropia_ plantada.
- 2.256 CP plantadas e 1.226 CP mapeadas, com taxa global de 54,3%.
- 831 detecções YOLO dentro das parcelas.
- 961 máscaras SAM3 por prompt textual e 267 máscaras SAM3 por bbox prompt.
- Categorias finais: 564 YOLO+SAM3, 267 só YOLO e 395 só SAM3.
- 1.228 copas segmentadas, com área média de 5,55 m² e mediana de 4,29 m².

## Ainda pendente

- Métricas de detecção no teste: Precisão, Recall, F1, mAP50, mAP50-95.
- Curva Precision-Recall e curvas de treino/validação.
- Importância dos hiperparâmetros.
- Figuras qualitativas de detecção e segmentação sobre o ortomosaico.
