# Estrutura Obrigatória do Relatório

## 1. Introdução

- Contexto do problema: detecção de árvores Cecropia (Embaúba) em mosaico aéreo
- Objetivo do trabalho
- Abordagem escolhida (visão computacional clássica, sem deep learning)
- Justificativa da abordagem

## 2. Dataset

- Número de tiles e dimensões (ex: 992 tiles, 2048×2048 px)
- Sobreposição entre tiles (ex: 512 px / 25%)
- Sistema de coordenadas e resolução espacial
- Tamanho aproximado em disco
- Descrição de casos especiais (ex: tiles completamente pretos, fora da área mapeada)
- Descrição do conjunto `data/validacao/`: organização por pasta, JSONs de
  revisão, labels `embauba`/`lixo` e caixas `faltantes`

## 3. Pipeline

Para **cada etapa** do pipeline, descrever:

- **O que faz**: descrição objetiva da operação
- **Motivação**: por que essa etapa foi escolhida — qual problema ela resolve no contexto da detecção de embaúba
- **Parâmetros**: valores utilizados e como foram determinados (empiricamente, literatura, etc.)

### Etapas obrigatórias a cobrir

| Etapa | Operação |
|-------|----------|
| 1 | Suavização (Gaussian Blur) |
| 2 | Conversão de espaço de cor (BGR → HSV) |
| 3 | Limiarização por cor (faixa HSV das copas) |
| 4 | Fechamento morfológico (conecta fragmentos) |
| 5 | Abertura morfológica (remove ruído) |
| 6 | Detecção de contornos |
| 7 | Filtragem por área mínima |
| 8 | Filtro final de embaúba (área/circularidade) |
| 9 | Convex Hull (aproximação da copa) |

## 4. Resultados

- Total de tiles processados e percentual com detecções
- Número total de detecções e área total estimada
- Tempo de processamento (total, médio por tile, mínimo e máximo)
- Estatísticas de detecções por tile (média, desvio padrão, máximo)
- Estatísticas de área por região detectada (média, mediana, máxima)
- Top 10 tiles por número de detecções (com arquivo, detecções, área e cobertura)
- Figuras geradas com breve legenda de cada uma

## 5. Conclusão

- Síntese dos resultados obtidos
- Limitações do método (falsos positivos, dependência de iluminação, sobreposição de tiles, etc.)
- Possíveis melhorias ou trabalhos futuros
