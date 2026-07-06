# Detecção e Segmentação de Copas de *Cecropia* (Embaúba) em Ortomosaico de VANT

Projeto final da disciplina **INE410159 — Visão Computacional** (UFSC).

Detecção e segmentação de copas de **Cecropia pachystachya** (embaúba) sobre um
único ortomosaico de VANT do experimento MataDIV, por **duas abordagens
complementares** avaliadas sobre a mesma área:

- **Parte I — Visão computacional clássica** (este repositório): segmentação por
  cor no espaço HSV, morfologia, contornos e filtros geométricos. Interpretável,
  leve e sem *deep learning* (apenas `cv2`, `numpy`, `matplotlib`).
- **Parte II — Redes neurais (SAM 3 + YOLOv8)**: pipeline semi-automatizado que
  usa o SAM 3 para gerar o dataset e segmentar as copas e o YOLOv8x para detectar
  os indivíduos. Documentado em [`docs/rede_neural/`](docs/rede_neural/).

## Grupo 10

- Augusto de H. V. Guerner
- Eduardo G. de Lazari
- Fábio H. A. Coelho
- Gabriel Arantes F. Gualda
- Nena A. Impanta

---

## Estrutura do repositório

| Caminho | Descrição |
|---------|-----------|
| `src/` | Código da abordagem clássica — ver [`src/index.md`](src/index.md) |
| `docs/relatorio/` | Relatório em LaTeX (`relatorio.pdf`) — ver [`docs/relatorio/index.md`](docs/relatorio/index.md) |
| `docs/apresentacao/` | Slides em LaTeX (`apresentacao.pdf`) e versão editável `apresentacao.pptx` (fonte Roboto) |
| `docs/rede_neural/` | Método e resultados da abordagem neural (SAM 3 + YOLOv8) |
| `data/tiles/` | 992 tiles JPEG 2048×2048 px (não versionados — ver link de download) |
| `data/validacao/` | Conjunto rotulado de validação — ver [`data/validacao/index.md`](data/validacao/index.md) |
| `data/output/` | Artefatos gerados pelo pipeline (não versionados) |
| `dev.sh` | Script único: roda o pipeline, a UI de validação ou compila os documentos |

Os tiles (~1,4 GB) não são versionados.
Download: <https://drive.google.com/drive/folders/1Gux57VofI_bxgt66hlXgaVlp_s21emNl>

---

## Como executar

```bash
./dev.sh pipeline            # pipeline completo nos 992 tiles → data/output/
./dev.sh anotar tile_0905    # UI OpenCV de validação (corrige a saída do detector)
./dev.sh figuras             # figuras qualitativas do relatório
./dev.sh relatorio           # compila docs/relatorio/relatorio.pdf
./dev.sh apresentacao        # compila docs/apresentacao/apresentacao.pdf
```

Uso direto do código (caminhos resolvidos a partir de `__file__`, rodam de
qualquer diretório):

```bash
pip install opencv-python matplotlib numpy

python3 src/main.py                            # pipeline completo → data/output/
python3 src/main.py <imagem_ou_pasta>          # detecção avulsa → pasta por imagem (JSON + figuras)
python3 src/main.py --validacao                # imprime as métricas do conjunto validado
python3 src/anotar.py tile_0905                 # valida uma entrada
python3 src/anotar.py "tile_01*" --amostra 20   # valida 20 tiles sorteados de um glob
python3 src/anotar.py data/validacao --resumo   # métricas do conjunto validado
```

Saídas de `src/main.py` sem path: uma pasta por tile em `data/output/tiles/tile_XXXX/`
(JSON de detecções, `grid.png` com os 10 painéis e cada etapa como PNG), os
histogramas/figuras estatísticas e o `relatorio.md`.

---

# Parte I — Visão Computacional Clássica

## Pipeline

```
Tile → Gaussian Blur → Segmentação HSV → CLOSE → OPEN → Contornos → Filtro (área/forma) → Convex Hull
```

| Etapa | Operação | Parâmetros | Motivação |
|-------|----------|------------|-----------|
| 1 | Gaussian Blur | kernel 9×9 | reduz ruído de textura antes da segmentação |
| 2 | Conversão BGR→HSV | `cv2.COLOR_BGR2HSV` | separa matiz/saturação/brilho, mais robusto à iluminação |
| 3 | Segmentação HSV | H:[44,58] S:[144,231] V:[104,203] | isola o verde-claro característico das copas |
| 4 | Morfologia CLOSE | elipse 25×25 | conecta fragmentos de uma mesma copa |
| 5 | Morfologia OPEN | elipse 9×9 | remove respingos de ruído |
| 6 | Detecção de contornos | `RETR_EXTERNAL` | transforma a máscara em objetos mensuráveis |
| 7 | **Filtro de área e forma** | **10.000 < área ≤ 600.000, (área≥33.798 OU circ≤0,155), solidez≥0,48** | **descarta manchas, palmeiras e blobs fundidos** |
| 8 | Convex Hull | `cv2.convexHull` | desenha o contorno final da copa |

### O filtro de área e forma (contribuição deste trabalho)

Uma segmentação HSV permissiva mantém **todo** contorno com área > 10.000 px², o
que gera muitos falsos positivos. Analisando detecções rotuladas, observou-se que:

- **palmeiras** marcam o *centro* de outra árvore → área pequena;
- **copas de embaúba** são grandes ou espalhadas e irregulares → baixa circularidade.

Logo, um contorno é aceito como embaúba quando é **grande** (`área ≥ 33.798 px²`)
**ou** pouco **circular** (`circularidade ≤ 0,155`), desde que a área não ultrapasse
`600.000 px²` (remove blobs fundidos gigantes) e a solidez seja `≥ 0,48` (evita
regiões excessivamente fragmentadas).

## Dataset

- **992 tiles** JPEG de 2048×2048 px de um ortomosaico georreferenciado.
- CRS **EPSG:31982** (UTM 22S), resolução ~0,00848 m/px (≈ 0,85 cm/pixel).
- Tiles completamente pretos (fora da área mapeada) são ignorados automaticamente
  (402 dos 992).

## Validação (`data/validacao/`)

Conjunto revisável para comparar versões do detector. Cada tile validado fica em
sua própria pasta:

```text
data/validacao/<nome>/
├── <nome>.jpg        # tile original (não versionado — pesado)
├── <nome>.json       # saída do detector + correções da revisão
└── <nome>_vis.png    # overlay: verde = embaúba, vermelho = lixo, azul = faltante
```

| Campo do JSON | Papel |
|---------------|-------|
| `embaubas` | detecções do detector (`bbox`, `area`, `poligono`) |
| `falsos_positivos` | índices de `embaubas` marcados como erro na revisão |
| `faltantes` | embaúbas visíveis que o detector não capturou |

A **existência** do JSON marca a entrada como validada (não há flag de revisão).
De `embaubas`/`falsos_positivos`/`faltantes` derivam-se TP/FP/FN. Esquema completo
em [`data/validacao/index.md`](data/validacao/index.md).

## Resultados (abordagem clássica)

Processamento completo dos 590 tiles válidos:

| Métrica | Valor |
|---------|-------|
| Tiles processados | 590 (402 pretos ignorados) |
| Tiles com ≥1 detecção | 507 (85,9%) |
| Total de detecções | 2.003 |
| Detecções por tile | 3,39 ± 2,69 |
| Área total estimada | 13.164,42 m² |
| Área média por região | 91.406 px² (≈ 6,57 m²) |
| Área máxima por região | 599.406 px² (≈ 43,10 m²) |
| Cobertura HSV média | 13,862% |
| Tempo médio por tile | ~400 ms (CPU, 4 workers) |
| Memória por tile | ~64 MB |

Validação manual sobre **50 tiles** revisados:

| Métrica | Valor |
|---------|-------|
| Detecções avaliadas | 207 |
| Verdadeiros positivos (TP) | 179 |
| Falsos positivos (FP) | 28 |
| Falsos negativos / faltantes (FN) | 79 |
| **Precisão** | **86,5%** |
| **Recall** | **69,4%** |
| **F1** | **77,0%** |

A precisão alta reflete o filtro final removendo bem os falsos positivos; o recall
mostra que parte das copas reais ainda escapa, principalmente quando a segmentação
HSV não gera candidato inicial.

---

# Parte II — Redes Neurais (SAM 3 + YOLOv8)

Pipeline semi-automatizado em seis etapas: (1) geração do dataset com **SAM 3**
(*text prompt* "tree") + curadoria por exclusão; (2) divisão espacial do dataset
(evita *data leakage* via clusterização por grafo + Monte Carlo); (3) correção
semi-automatizada das anotações; (4) treinamento do **YOLOv8x** (otimização
bayesiana com Optuna/TPE); (5) inferência e pós-processamento (duas rodadas +
deduplicação); (6) segmentação das copas novas com SAM 3 (*bbox prompt*).

O dataset partiu de 4.569 polígonos gerados pelo SAM 3, reduzidos por curadoria a
**1.109 máscaras** de *Cecropia*, que geraram 7.488 instâncias de treino (7.786
após correção).

## Resultados (abordagem neural)

Consolidação por parcela (144 parcelas, 60 com CP plantada, 2.256 indivíduos
plantados):

| Métrica | Valor |
|---------|-------|
| Indivíduos mapeados | 1.226 |
| Taxa global de detecção | 54,3% |
| Copas segmentadas | 1.228 |
| Área média da copa | 5,55 m² (mediana 4,29 m²) |

Categorias do mapeamento final:

| Categoria | Quantidade | Proporção |
|-----------|-----------:|----------:|
| YOLO + SAM 3 | 564 | 46,0% |
| Só YOLO | 267 | 21,8% |
| Só SAM 3 | 395 | 32,2% |

Melhores taxas por composição: D12 (68,3%), D7 (65,9%), D8 (65,7%); a monocultura
D1 concentrou o maior desafio (36,8%). Detalhes e figuras em
[`docs/rede_neural/`](docs/rede_neural/).

---

## Comparação das abordagens

| Aspecto | Clássica (HSV) | Neural (SAM 3 + YOLOv8) |
|---------|----------------|-------------------------|
| Sinal | Cor + forma | Padrões aprendidos |
| Anotação | Não requer | Curadoria por exclusão |
| Interpretabilidade | Alta | Baixa |
| Custo | Baixo (~400 ms/tile, CPU) | Alto (GPU, 500 épocas) |
| Segmentação | Fecho convexo | Máscara SAM 3 |
| Resultado | P 86,5% / R 69,4% / F1 77,0% | 1.226 mapeadas; 54,3% global |

A clássica serve de *baseline* interpretável e rápido; a neural busca precisão e
generalização ao custo de infraestrutura e treinamento.

---

## Uso de Inteligência Artificial

- **IA como método** (Parte II): SAM 3 (Meta AI) para geração do dataset e
  segmentação das copas; YOLOv8x (Ultralytics) para detecção; Optuna (TPE) na
  otimização de hiperparâmetros.
- **IA como assistente de desenvolvimento**: o assistente Claude (Anthropic) foi
  usado para refatorar o código do pipeline clássico, gerar as figuras e redigir e
  revisar o relatório e os slides, com revisão e responsabilidade final dos autores.

---

## Arquivos principais

| Arquivo | Descrição |
|---------|-----------|
| `src/core/deteccao.py` | Núcleo: HSV/morfologia/hull, filtro de área e forma e `analisar` (bounds + métricas) |
| `src/core/execucao.py` | Processamento paralelo (`multiprocessing.Pool`) + checkpoint/retomada |
| `src/core/estatisticas.py` | Figuras estatísticas e geração do `relatorio.md` |
| `src/main.py` | Entrada principal: pipeline completo ou detecção por imagem/pasta |
| `src/anotar.py` | UI OpenCV de validação (falsos positivos + faltantes) |
