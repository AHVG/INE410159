# Detecção de Embaúba (Cecropia) — Visão Computacional Clássica

Projeto final da disciplina de Visão Computacional — UFSC.

Este trabalho faz parte de uma comparação entre duas abordagens para detecção e
segmentação de copas de **Cecropia pachystachya** em ortomosaicos de VANT:

- **Visão computacional clássica**: abordagem deste repositório, baseada em HSV,
  morfologia, contornos e filtros geométricos.
- **SAM 3 e YOLOv8**: outra frente do trabalho, usando modelos modernos para
  detecção e segmentação.

## Grupo 10

- Gabriel Arantes F. Gualda
- Fábio H. A. Coelho
- Nena A. Impanta
- Augusto de H. V. Guerner
- Eduardo G. de Lazari

Detecta árvores **Cecropia sp. (Embaúba)** em um mosaico aéreo usando segmentação
por cor no espaço HSV e morfologia, **sem deep learning** (apenas `cv2`, `numpy`,
`matplotlib`).

O projeto parte do script base do professor (`embaubaHSVmask.py`) e adiciona um
**filtro de pós-processamento** que reduz drasticamente os falsos positivos
(centros de palmeira e manchas de mato confundidos com embaúba), além de um
conjunto de **validação** revisável para comparar versões do pipeline.

---

## Pipeline

```
Tile → Gaussian Blur → Segmentação HSV → CLOSE → OPEN → Contornos → Filtro (área/circ) → Convex Hull
```

| Etapa | Operação | Parâmetros | Motivação |
|-------|----------|------------|-----------|
| 1 | Gaussian Blur | kernel 9×9 | reduz ruído de textura antes da segmentação |
| 2 | Segmentação HSV | H:[44,58] S:[144,231] V:[104,203] | isola o verde característico das copas |
| 3 | Morfologia CLOSE | elipse 25×25 | conecta fragmentos de uma mesma copa |
| 4 | Morfologia OPEN | elipse 9×9 | remove respingos de ruído |
| 5 | **Filtro forma/tamanho** | **10.000 < área ≤ 600.000, (área≥33.798 OU circ≤0.155), solidez≥0.48** | **descarta manchas, palmeiras e blobs fundidos** |
| 6 | Convex Hull | — | desenha o contorno da copa |

### O filtro `eh_embauba` (contribuição deste trabalho)

O baseline mantém **todo** contorno com área > 10.000 px², o que gera muitos
falsos positivos. Analisando detecções rotuladas, encontramos que:

- **palmeiras** marcam o *centro* de outra árvore → área pequena;
- **copas de embaúba** são grandes ou espalhadas e irregulares → baixa circularidade.

Logo, um contorno é aceito como embaúba quando é **grande** (`área ≥ 33.798 px²`)
**ou** pouco **circular** (`circularidade ≤ 0.155`).

> A densidade na máscara bruta (`fill`) também foi testada como feature, mas foi
> **descartada**: ela melhora alguns casos rotulados, mas tende a descartar
> embaúbas grandes e esparsas.

---

## Dataset

- **992 tiles** JPEG de 2048×2048 px, com 25% (512 px) de sobreposição.
- Mosaico aéreo georreferenciado em **EPSG:31982** (UTM 22S), ~0.00848 u/px.
- Tiles completamente pretos (fora da área mapeada) são ignorados automaticamente.

### Conjunto de validação (`data/validacao/`)

Para comparar versões do detector, **153 detecções** do baseline (em **18 tiles**)
foram rotuladas manualmente como `embauba` ou `lixo`. O esquema completo dos
JSONs está em [`data/validacao/index.md`](data/validacao/index.md).

| Tiles | Detecções | Embaúba | Lixo | Uso |
|-------|-----------|---------|------|-----|
| 18 | 153 | 41 | 112 | validação comparativa e revisão manual |

Além dos candidatos rotulados, cada JSON pode registrar `faltantes`: caixas de
embaúbas visíveis no tile que o detector não encontrou.

---

## Validação

O conjunto em `data/validacao/` é a base revisável para comparar versões do
pipeline. Ele guarda os candidatos do pipeline base, os rótulos manuais
`embauba`/`lixo` e caixas `faltantes` para copas visíveis que não viraram
candidato.

Cada imagem de validação fica isolada em sua própria pasta:

```text
data/validacao/<nome>/
├── <nome>.jpg
├── <nome>.json
└── <nome>_vis.png
```

**Uso esperado:**

| Forma | Papel |
|-------|-------|
| `embaubas` | detecções do detector (`bbox`, `area`, `poligono`) |
| `falsos_positivos` | índices de `embaubas` marcados como erro na validação |
| `faltantes` | embaúbas visíveis que o detector não capturou |

A **existência** do JSON em `data/validacao/<tile>/` marca a entrada como
validada — não há flag de revisão. O conjunto validado é a referência para as
métricas do relatório e para comparar pipelines.

---

## Instalação

```bash
pip install opencv-python matplotlib numpy
```

## Uso

O código fica em `src/` e os dados em `data/`. As entradas públicas por ora são
`src/main.py` e `src/anotar.py`; ambas compartilham o núcleo
`core/deteccao.py`. Os caminhos de dados são resolvidos a partir de `__file__`,
então os comandos rodam de qualquer diretório:

```bash
./run.sh                                     # pipeline completo (atalho)
MPLBACKEND=Agg python3 src/main.py           # pipeline completo nos 992 tiles → data/output/
python3 src/main.py <imagem_ou_pasta>         # detecção avulsa → pasta por imagem (JSON + figuras)
python3 src/anotar.py tile_0905              # valida uma entrada (corrige a saída)
python3 src/anotar.py "tile_01*" --amostra 20 # valida 20 tiles sorteados de um glob
python3 src/anotar.py data/validacao --resumo # métricas do conjunto validado
```

### Saídas

- `src/main.py` sem path → `data/output/`: uma pasta por tile em `tiles/tile_XXXX/`
  (JSON de detecções + `grid.png` + as 10 etapas do pipeline), histogramas e `relatorio.md`.
- `src/main.py <imagem_ou_pasta>` → `<path>_deteccoes/` com uma **pasta por imagem**
  (`<nome>/<nome>.json` com `bbox`, `area`, `poligono` + `grid.png` + as 10 etapas).
  Use `--saida` para outro destino.
- `src/anotar.py <seleção>` → UI OpenCV que abre, uma por uma, só as entradas
  **ainda não validadas** com as detecções do detector já desenhadas: clique
  marca falso positivo, arrasto desenha uma embaúba faltante. Salvar grava
  `data/validacao/<nome>/` (`<nome>.jpg`, `<nome>.json`, `<nome>_vis.png`) — a
  existência do JSON já marca como validada. A seleção aceita vários tiles,
  caminhos, globs ou uma pasta, com `--amostra N` para sortear um subconjunto;
  `--refazer`/`--incluir-validadas` para reabrir as já feitas e `--resumo` para
  as métricas.

A saída canônica por imagem é `core.deteccao.analisar(img)`, que devolve
`embaubas` (uma entrada por copa — `bbox`, `area`, `poligono` — depois do filtro)
mais as métricas de tempo/memória. É o que alimenta os JSONs do pipeline e o
estado inicial da validação manual no `anotar.py`, onde você marca
`falsos_positivos` e `faltantes`.

---

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `src/ref/embaubaHSVmask.py` | Script base original do professor (referência — **não modificar**) |
| `src/core/deteccao.py` | **Núcleo**: HSV/morfologia/hull, filtro `filtro_forma_tamanho` e `analisar` (bounds + métricas) |
| `src/core/execucao.py` | Processamento paralelo (`multiprocessing.Pool`) + checkpoint/retomada |
| `src/core/estatisticas.py` | Figuras estatísticas e geração do `relatorio.md` |
| `src/main.py` | Entrada principal: output completo sem path ou JSON por imagem/pasta |
| `src/anotar.py` | UI OpenCV de validação: corrige a saída do detector (falsos positivos + faltantes) |
| `data/tiles/` | 992 tiles JPEG (não versionados — ver link de download) |
| `data/validacao/` | Conjunto rotulado de validação (ver `data/validacao/index.md`) |
| `data/output/` | Saídas geradas pelo pipeline (não versionado) |
| `index.md` | Estrutura obrigatória do relatório |
