# Solução com Redes Neurais (SAM 3 + YOLOv8) — Material-fonte

> Transcrição editável do relatório da frente de redes neurais
> (`sam3_yolo_detection.pdf`). Serve de **entrada** para a geração do relatório
> final combinado em [`../relatorio/relatorio.tex`](../relatorio/relatorio.tex).
>
> **Estado:** o material original termina em "3.3.7 Métricas de avaliação".
> **Não há Resultados, Discussão nem Conclusão** — ver a seção "Pendências" no
> final e o [`index.md`](index.md) desta pasta.
>
> Marcadores `[FIGURA X]` / `[TABELA X]` do original foram mantidos e mapeados
> para arquivos esperados em [`figuras/`](figuras/) e [`resultados/`](resultados/).

---

## 1. Introdução

O conhecimento detalhado sobre árvores individuais — identificação, distribuição
espacial e desenvolvimento ao longo do tempo — constitui base fundamental para a
pesquisa botânica, o manejo florestal e a conservação da biodiversidade (Wielgosz
et al., 2024; Fassnacht et al., 2016). Tradicionalmente, essas informações são
obtidas por inventários de campo que, embora eficazes, apresentam custos elevados,
complexidade logística e demanda por mão de obra especializada (Ferretti et al.,
2024). A limitação torna-se mais relevante diante da Década da Restauração de
Ecossistemas (2021–2030), que amplia a demanda por ferramentas de acompanhamento
eficiente de áreas florestais (De Almeida et al., 2020).

Os Veículos Aéreos Não Tripulados (VANTs) consolidaram-se como ferramenta de alta
relevância para o monitoramento florestal: imagens com resolução espacial na ordem
de centímetros viabilizam a observação de características morfológicas individuais
das plantas — formato e coloração das copas, ramificação e textura foliar — em
nível inviável com satélite convencional (Soltani et al., 2025; Sun et al., 2021).
Contudo, a análise visual de ortomosaicos por especialistas torna-se inviável em
dezenas ou centenas de hectares. Técnicas de aprendizado profundo têm sido cada
vez mais aplicadas: CNNs aprendem padrões visuais diretamente dos exemplos, sem
que o pesquisador precise definir manualmente as características que distinguem uma
espécie da outra (LeCun et al., 1998; Wagner et al., 2019), com resultados
promissores para o mapeamento de copas em diferentes biomas (Schiefer et al.,
2020; Ferreira et al., 2020; Braga et al., 2020).

Apesar dos avanços, a aplicação prática de modelos supervisionados esbarra na
necessidade de grandes volumes de dados de treinamento anotados. Em ambientes
florestais com espécies morfologicamente semelhantes sob dosséis densos, a
anotação manual é a etapa mais demorada e custosa de todo o processo (Kattenborn
et al., 2021; Galuszynski et al., 2022).

Modelos fundacionais de segmentação oferecem alternativa: o **Segment Anything
Model (SAM)**, da Meta AI, é treinado em mais de um bilhão de máscaras e segmenta
objetos em domínios para os quais nunca foi especificamente treinado, como
ortomosaicos de VANT (Kirillov et al., 2023). Sua terceira versão, o **SAM 3**,
introduziu a segmentação a partir de *prompts textuais*: ao receber a palavra
"tree", delineia exclusivamente as copas de árvores, ignorando os demais elementos
da paisagem. Isso permite a **"curadoria por exclusão"**: em vez de desenhar cada
anotação, o pesquisador filtra um conjunto pré-existente de candidatos, removendo
os que não correspondem à espécie-alvo.

A espécie-alvo é a **_Cecropia pachystachya_ Trécul (Urticaceae)**, embaúba —
árvore pioneira de rápido crescimento, indicadora do estágio de regeneração, com
copa umbeliforme e folhas palmatilobadas de coloração verde-prateada (Berg et al.,
2005; Carvalho, 2008; Wagner et al., 2019). O estudo foi conduzido no experimento
**MataDIV** (Mata Atlântica Diversidade), da Estação Experimental de Itatinga
(ESALQ/USP), que manipula a riqueza de espécies em 144 parcelas.

**Questão de pesquisa:** é viável construir um pipeline de detecção e segmentação
de copas a partir de ortomosaicos de VANT, utilizando o SAM 3, com poucas etapas
manuais e replicável para diferentes espécies arbóreas?

## 2. Objetivos

### 2.1 Objetivo geral

Desenvolver e avaliar uma metodologia para detecção e segmentação individual de
copas de _Cecropia pachystachya_ em ortomosaico de VANT, integrando SAM 3 e
YOLOv8, com ênfase na redução do esforço manual.

### 2.2 Objetivos específicos

- Desenvolver um pipeline semi-automatizado de detecção e segmentação com SAM 3,
  com o menor número possível de etapas manuais.
- Avaliar o desempenho na detecção e segmentação de copas de _C. pachystachya_ em
  diferentes cenários de composição florestal do MataDIV.
- Discutir a replicabilidade do pipeline para outras espécies arbóreas.

## 3. Materiais e Métodos

### 3.1 Área de estudo

Experimento MataDIV, Estação Experimental de Ciências Florestais de Itatinga
(ESALQ/USP), Itatinga–SP (23°06'S, 48°39'W). Experimento de diversidade arbórea e
funcionamento ecossistêmico (BEF) da rede TreeDivNet, plantado em 2019 em antigo
talhão de _Eucalyptus_ (Verheyen et al., 2016; Paquette et al., 2018). Manipula
três variáveis (riqueza de espécies, fertilização, disponibilidade hídrica) em 144
parcelas (~8 ha), cada uma de 20 m × 23 m com 100 árvores em espaçamento de 2,3 m
(~14.400 indivíduos). Seis espécies nativas de estratégias contrastantes em 12
estratos (de monoculturas a alta diversidade com 20 espécies). _C. pachystachya_
está em cinco composições, de 100 indivíduos/parcela na monocultura a ~5 nas de
alta diversidade.

> `[FIGURA X — Mapa de localização: Brasil, São Paulo, Itatinga, MataDIV]`
> → `figuras/mapa_localizacao.png`
>
> `[TABELA 1 — Composições de espécies no experimento MataDIV]`
> → `figuras/tabela1_composicoes.*`

### 3.2 Coleta de dados

Sobrevoo em **13 de março de 2026** com VANT **DJI Matrice 4T**, câmera RGB wide
(CMOS 1/1.3", 48 MP, focal equivalente 24 mm, FOV 82°), a 23 m de altitude, com
sobreposição frontal e lateral de 80%, **GSD ≈ 0,85 cm/pixel**. Fotogrametria no
**Agisoft Metashape** → ortomosaico de **48.264 × 47.029 px**, 4 bandas (RGBA),
~8,46 GB em GeoTIFF, **SIRGAS 2000 / UTM 22S (EPSG:31982)**.

### 3.3 Métodos

Pipeline semi-automatizado em **seis etapas** que integra SAM 3 e YOLOv8:
(1) geração do dataset com SAM 3 text prompt; (2) divisão espacial; (3) correção
das anotações; (4) treinamento YOLOv8x; (5) inferência e pós-processamento;
(6) segmentação com SAM 3 bbox prompt.

> `[FIGURA X — Fluxograma geral do pipeline semi-automatizado (6 etapas)]`
> → `figuras/fluxograma_pipeline.png`
>
> `[FIGURA X — Exemplos de copas de C. pachystachya no ortomosaico: (a) copa
> isolada, (b) copas adjacentes em monocultura, (c) copa jovem]`
> → `figuras/exemplos_copas.png`

#### 3.3.1 Geração do dataset de treinamento

Ortomosaico recortado em tiles de **4.096 × 4.096 px** (overlap 1.024 px) e
submetido ao SAM 3 com prompt textual **"tree"** → **4.569 polígonos** de diversas
espécies. Curadoria por exclusão em **QGIS** → restaram **1.109 máscaras** de
_C. pachystachya_ (975 dentro do MataDIV, 134 no entorno). Bounding boxes geradas
automaticamente das máscaras (geometria envolvente), sem traçado manual.

As máscaras têm dupla função: base para o dataset de detecção **e** produto final
de segmentação para os indivíduos desta fase (as 975 cecropias dentro do MataDIV
já ficam segmentadas).

Para o detector, ortomosaico recortado em tiles de **896 × 896 px** (overlap 50%).
Tamanho definido pela maior copa identificada. Overlap de 50% funciona como data
augmentation implícito → 1.109 indivíduos geraram **7.488 instâncias** (fator médio
6,8×). Coordenadas convertidas para formato YOLO (vértices normalizados 0–1).

#### 3.3.2 Divisão espacial do dataset

Divisão em treino (70%) / validação (15%) / teste (15%) **não** aleatória, para
evitar *data leakage* (Roberts et al., 2017; Kattenborn et al., 2022). Algoritmo
de clusterização espacial em quatro etapas:

1. Cada anotação é associada ao tile de centroide mais próximo.
2. Anotações que intersectam >20% da área em mais de um tile vinculam os tiles.
3. Vínculos formam um grafo; componentes conexos via BFS → clusters de tiles.
4. Clusters distribuídos por otimização Monte Carlo (1.000 iterações), minimizando
   o MSE entre proporções obtidas e metas 70/15/15.

#### 3.3.3 Correção semi-automatizada das anotações

O 1º ciclo de treino revelou falsos negativos que eram cecropias reais ausentes
das anotações do SAM 3. O modelo foi aplicado a validação/teste e cada detecção
comparada por **IoU**; detecções com IoU < 0,5 = potenciais indivíduos não
anotados. Visualização com anotações existentes (verde) e detecções sem match
(vermelho) para curadoria do operador.

> `[FIGURA X — Anotações existentes (verde) e detecções sem correspondência (vermelho)]`
> → `figuras/correcao_anotacoes.png`

- **Teste:** das 163 detecções sem match, 157 confirmadas, 6 descartadas.
- **Validação:** 144 novas anotações incorporadas, 2 descartadas.

Correção aplicada só a validação/teste (~160 imagens cada); pode ser estendida ao
treino (776 imagens).

**[TABELA 2 — Estrutura final do dataset]**

| Conjunto     | Imagens | Instâncias (original) | Instâncias (corrigido) |
|--------------|--------:|----------------------:|-----------------------:|
| Treinamento  |     776 |                 6.300 |     6.300 (sem alter.) |
| Validação    |     167 |                   543 |            687 (+144)  |
| Teste        |     166 |                   645 |            799 (+154)  |
| **Total**    |   1.109 |                 7.488 |                  7.786 |

#### 3.3.4 Treinamento do modelo de detecção

**YOLOv8x** (Ultralytics; Jocher et al., 2023), versão de detecção (não segmentação
— o SAM 3 delineia as copas). Otimização bayesiana com **Optuna** (Akiba et al.,
2019) e amostrador **TPE**. Espaço de busca: **21 hiperparâmetros** em três
categorias. Importância via regressor **Random Forest** (500 árvores, Mean Decrease
Impurity). Otimização: 30 tentativas × 20 épocas, mAP50 na validação, batch 8,
imagem 896 px. **Treino final: 500 épocas, imagem 896 px, batch 8, AdamW.**

**[TABELA X — Espaço de busca dos hiperparâmetros otimizados]**

| Categoria | Hiperparâmetro | Funcionalidade | Inf. | Sup. |
|-----------|----------------|----------------|-----:|-----:|
| Otimizador | Freeze Layers | Camadas sem atualização de pesos | 0 | 21 |
| Otimizador | Learning Rate (lr0) | Taxa de aprendizagem inicial | 1e-05 | 0,01 |
| Otimizador | Final LR Factor (lrf) | Fator de decaimento final do LR | 0,01 | 1,0 |
| Otimizador | Momentum | Inércia do otimizador | 0,6 | 0,98 |
| Otimizador | Weight Decay | Penaliza pesos grandes (overfitting) | 0,0001 | 0,001 |
| Otimizador | Warmup Epochs | Épocas iniciais com LR crescente | 0 | 5 |
| Perda | Box Loss Weight | Peso da perda de localização da bbox | 0,02 | 0,2 |
| Perda | Class Loss Weight | Peso da perda de classificação | 0,2 | 4,0 |
| Augment. | HSV Hue Aug. | Alteração da tonalidade | 0 | 0,1 |
| Augment. | HSV Sat. Aug. | Alteração da saturação | 0 | 0,9 |
| Augment. | HSV Val. Aug. | Alteração do brilho | 0 | 0,9 |
| Augment. | Rotation (deg) | Rotação | 0 | 45 |
| Augment. | Translation | Translação | 0 | 0,9 |
| Augment. | Scale Aug. | Escala | 0 | 0,9 |
| Augment. | Shear (deg) | Cisalhamento | 0 | 10 |
| Augment. | Perspective | Perspectiva | 0 | 0,001 |
| Augment. | Flip Up-Down | Espelhamento vertical | 0 | 1 |
| Augment. | Flip Left-Right | Espelhamento horizontal | 0 | 1 |
| Augment. | Mosaic | Combinação de 4 imagens | 0 | 1 |
| Augment. | MixUp | Sobreposição de 2 imagens | 0 | 1 |
| Augment. | Copy-Paste | Cópia de segmentos entre imagens | 0 | 1 |

#### 3.3.5 Inferência e pós-processamento

Modelo aplicado ao ortomosaico completo (tiles 896 px, overlap 50%) em **duas
rodadas**: grade regular + tiles deslocados em meio passo (offset 50%), para cobrir
bordas.

> `[FIGURA X — Duas rodadas de inferência: grade regular vs. deslocada]`
> → `figuras/rodadas_inferencia.png`

- Descarte de detecções a <10 px da borda.
- Deduplicação: pares com sobreposição >10% da menor área → mantém maior confiança.
- Filtros: descarta confiança <0,65; área <1,75 m² ou >40 m²; fora do MataDIV.
- Cruzamento com máscaras SAM 3 → 3 categorias: (i) YOLO + máscara prévia;
  (ii) só YOLO (sem máscara); (iii) só SAM 3 inicial (não detectado pelo YOLO).

> `[FIGURA X — Cruzamento espacial YOLO × SAM 3 (três categorias)]`
> → `figuras/cruzamento_categorias.png`

Categorias (i) e (iii) já têm segmentação; (ii) segue para a etapa 3.3.6.

#### 3.3.6 Segmentação das copas com SAM 3 por prompt geométrico

Para os indivíduos só do YOLO, SAM 3 com **prompt geométrico** (bbox como entrada).
Recorte ao redor da bbox com *padding*; SAM 3 gera máscaras candidatas; seleção
central (centroide mais próximo do centro). Máscaras → polígonos georreferenciados
→ limpeza morfológica.

> `[FIGURA X — Segmentação por prompt geométrico: (a) recorte+bbox, (b) máscara SAM 3, (c) polígono]`
> → `figuras/segmentacao_bbox_prompt.png`

**[TABELA X — Parâmetros da segmentação por prompt geométrico com SAM 3]**

| Parâmetro | Valor adotado |
|-----------|---------------|
| Padding ao redor da bbox | 300 cm |
| Área mínima do polígono | 0,3 m² |
| Confiança do SAM 3 | 0,25 |
| Estratégia de seleção | Central (centroide mais próximo do centro) |

Polígonos integrados às máscaras da segmentação inicial → mapeamento final.

#### 3.3.7 Métricas de avaliação

Avaliação no conjunto de teste; cada detecção = VP, FP ou FN. Correspondência por
**IoU** (limiar 0,5): `IoU = Área da Interseção / Área da União`.

- `P = VP / (VP + FP)` — Precision
- `R = VP / (VP + FN)` — Recall
- `F1 = 2·P·R / (P + R)`
- `AP = ∫₀¹ P(R) dR` — área sob a curva Precision-Recall

Como há uma única classe, **mAP = AP**; mantida a notação **mAP50** (padrão
Ultralytics). Usa-se também **mAP50-95** (média da AP em 10 limiares de IoU de 0,5
a 0,95).

> `[FIGURA X — Cálculo de IoU entre caixa predita e de referência]`
> → `figuras/iou_exemplo.png`

---

## Pendências (a preencher para o relatório final)

O material original **termina aqui**. Para completar a Parte II do relatório final,
faltam — colocar os artefatos em [`resultados/`](resultados/):

- [ ] **4. Resultados**: métricas de detecção no teste (Precisão, Recall, F1,
      mAP50, mAP50-95).
- [ ] Curvas Precision-Recall e curvas de treino/validação (loss, mAP por época).
- [ ] Importância dos hiperparâmetros (Random Forest / Mean Decrease Impurity).
- [ ] Contagem final de indivíduos mapeados por categoria (i / ii / iii).
- [ ] Figuras qualitativas de detecção e segmentação sobre o ortomosaico.
- [ ] **5. Discussão** e **6. Conclusão** da frente neural.
- [ ] Figuras do texto (todos os `[FIGURA X]` acima) em [`figuras/`](figuras/).
