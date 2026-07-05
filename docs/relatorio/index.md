# Estrutura do Relatório Final (Combinado)

O relatório final integra **duas soluções** para a detecção e segmentação de copas
de _Cecropia pachystachya_ sobre o **mesmo ortomosaico** de VANT (experimento
MataDIV):

- **Parte I — Visão Computacional Clássica** (HSV + morfologia + filtros), gerada a
  partir do pipeline deste repositório (`data/output/`).
- **Parte II — Redes Neurais (SAM 3 + YOLOv8)**, cujo material-fonte fica em
  [`../rede_neural/`](../rede_neural/).

Fonte LaTeX: [`relatorio.tex`](relatorio.tex). Compilar pela raiz com
`./dev.sh relatorio`.

> A Parte II já possui contagens por parcela/composição e estatísticas de área.
> Manter `\pendente{...}` apenas para métricas de teste, curvas, importância de
> hiperparâmetros e figuras qualitativas ainda ausentes.

---

## Ordem geral do documento

```
1. Introdução (unificada)
2. Objetivos
3. Área de Estudo e Coleta de Dados      ← comum às duas frentes

PARTE I — Solução com Visão Computacional Clássica
4. Dataset (recorte para a abordagem clássica)
5. Pipeline de Detecção Clássico (9 etapas)
6. Resultados da Abordagem Clássica
7. Interpretação e Limitações da Abordagem Clássica

PARTE II — Solução com Redes Neurais (SAM 3 + YOLOv8)
8. Metodologia da Abordagem Neural
9. Resultados da Abordagem Neural

PARTE III — Síntese
10. Discussão Comparativa
11. Conclusões
```

---

## Seção comum

### 1. Introdução
- Contexto: monitoramento florestal por VANT; importância da _Cecropia_ como
  indicadora de regeneração e sua distinguibilidade visual.
- As **duas frentes complementares** e por que operam sobre o mesmo ortomosaico.
- Objetivo e justificativa de comparar abordagem clássica × neural.

### 2. Objetivos
- Objetivo geral e objetivos específicos das duas frentes.

### 3. Área de Estudo e Coleta de Dados
- Experimento MataDIV (Itatinga/ESALQ-USP): parcelas, composições de espécies.
- Sobrevoo e ortomosaico: VANT, GSD ≈ 0,85 cm/px, dimensões, CRS (EPSG:31982).
- Deixar explícito que é a **base de dados comum**; cada parte descreve seu recorte.

---

## PARTE I — Visão Computacional Clássica

### 4. Dataset (recorte clássico)
- 992 tiles, 2048×2048 px, sobreposição, CRS e resolução, tamanho em disco.
- Tiles pretos (fora da área) ignorados; tiles processados.
- Conjunto `data/validacao/`: organização por pasta, JSONs (`embaubas`,
  `falsos_positivos`, `faltantes`).

### 5. Pipeline de Detecção Clássico
Subseção de **motivação geral** (por que dividir em etapas clássicas: cor →
morfologia → contornos → filtros geométricos) e, depois, **cada etapa** em subseção
própria com imagem, motivação, o que faz e parâmetros. Preferir imagens do passo a
passo em `data/output/passo_a_passo/`.

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

### 6. Resultados da Abordagem Clássica
Ordem das subseções:
1. Estatísticas gerais (tiles processados/ignorados, % com detecção, total de
   detecções, área total, tempo total/médio, detecções por tile, área por região,
   memória média e pico).
2. Distribuição de detecções por tile (histograma + interpretação).
3. Distribuição das áreas de detecção (histograma + interpretação).
4. Cobertura HSV por tile (figura; explicar que mede pixels na faixa, não copas).
5. Top 10 tiles com mais detecções (figura e/ou tabela).
6. Validação manual (tiles revisados, TP/FP/FN, Precisão/Recall/F1 + interpretação).
7. Desempenho computacional (figuras de tempo e memória + comentário).

### 7. Interpretação e Limitações da Abordagem Clássica
- Taxa de tiles com detecções; falsos positivos por confusão espectral; regiões
  grandes/fundidas e limites dos filtros; leitura da validação.

---

## PARTE II — Redes Neurais (SAM 3 + YOLOv8)

Conteúdo transcrito de [`../rede_neural/relatorio.md`](../rede_neural/relatorio.md).

### 8. Metodologia da Abordagem Neural
- Contexto: SAM 3 e curadoria por exclusão.
- 3.3.1 Geração do dataset (SAM 3 text prompt; 1.109 máscaras; 7.488 instâncias).
- 3.3.2 Divisão espacial (clusterização por grafo/BFS + Monte Carlo).
- 3.3.3 Correção semi-automatizada (tabela de estrutura final do dataset).
- 3.3.4 Treinamento YOLOv8x (Optuna/TPE; tabela dos 21 hiperparâmetros).
- 3.3.5 Inferência e pós-processamento (duas rodadas; deduplicação; 3 categorias).
- 3.3.6 Segmentação com SAM 3 por prompt geométrico (tabela de parâmetros).
- 3.3.7 Métricas de avaliação (IoU, P, R, F1, AP, mAP50, mAP50-95).

### 9. Resultados da Abordagem Neural
- Contagem geral: 144 parcelas, 60 com CP plantada, 2.256 CP plantadas e 1.226 CP
  mapeadas.
- Taxa global de detecção: 54,3%; 1.228 copas segmentadas.
- Categorias finais: 564 YOLO+SAM3, 267 só YOLO e 395 só SAM3.
- Resultados por composição, com destaque para D12 (68,3%), D7 (65,9%), D8
  (65,7%) e D11 (62,7%).
- Estatísticas de área: média 5,55 m², mediana 4,29 m², máximo 24,26 m².
- Ainda pendente: métricas no teste, curvas P-R e de treino, importância dos
  hiperparâmetros e figuras qualitativas.

---

## PARTE III — Síntese

### 10. Discussão Comparativa
- Tabela comparando as duas abordagens (sinal explorado, anotação,
  interpretabilidade, custo, segmentação, discriminação de espécie, resultado).
- Como as frentes se complementam.

### 11. Conclusões
- Síntese dos resultados de cada frente e trabalhos futuros (clássica, neural e
  integração das duas).

---

## Arquivos das pastas de documentação

| Arquivo | Descrição |
|---------|-----------|
| `relatorio/relatorio.tex` | Fonte LaTeX do relatório final combinado |
| `relatorio/relatorio.pdf` | PDF compilado |
| `relatorio/index.md` | Este arquivo (estrutura do relatório) |
| `apresentacao/` | Slides finais (ver `apresentacao/index.md`) |
| `rede_neural/` | Material-fonte da frente neural (ver `rede_neural/index.md`) |

```bash
./dev.sh relatorio
./dev.sh apresentacao
```
