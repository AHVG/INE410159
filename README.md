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
| 5 | Filtro de contornos | área > 10.000 px² | descarta manchas minúsculas |
| 6 | **Filtro `eh_embauba`** | **área≥33.798 OU circ≤0.155** | **descarta palmeiras (pequenas/circulares)** |
| 7 | Convex Hull | — | desenha o contorno da copa |

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
| `deteccoes` | candidatos do pipeline base com label manual |
| `faltantes` | embaúbas visíveis não capturadas pelo pipeline base |
| `revisado` | indica que o tile foi conferido no `anotar.py` |

Quando todos os tiles estiverem `revisado: true`, o conjunto passa a ser uma
referência consistente para comparações futuras no relatório.

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
python3 src/main.py <imagem_ou_pasta>         # detecção avulsa → JSON por imagem
python3 src/main.py <imagem_ou_pasta> --vis --passo-a-passo
python3 src/anotar.py tile_0905              # revisão manual de labels/faltantes
python3 src/anotar.py caminho/nova_imagem.jpg # adiciona imagem nova à validação
python3 src/anotar.py data/validacao --pendentes
```

### Saídas

- `src/main.py` sem path → `data/output/`: passo a passo por tile, histogramas e `relatorio.md`.
- `src/main.py <imagem_ou_pasta>` → `<path>_deteccoes/` com um **JSON de bounds**
  por imagem (`<nome>.json`: `bbox`, `area`, `poligono`). Use `--vis` para também
  salvar o PNG anotado, `--passo-a-passo` para os 9 painéis do pipeline e `--saida` para outro destino.
- `src/anotar.py <tile>` → UI OpenCV para alternar `embauba`/`lixo`, desenhar
  caixas de `faltantes` e marcar o JSON como `revisado`. Também aceita
  `data/validacao` para revisar vários tiles em fila; use `--pendentes` para
  abrir só os ainda não revisados.
- `src/anotar.py <imagem>` → copia a imagem para `data/validacao/`, gera o JSON
  inicial com candidatos e abre a revisão. Cada imagem fica em sua própria pasta:
  `data/validacao/<nome>/<nome>.jpg`, `<nome>.json` e `<nome>_vis.png`.

A saída canônica do pipeline é `core.deteccao.detectar_embaubas(img) → lista de
bounds` (uma entrada por copa, depois do filtro). Para validação manual,
`core.deteccao.extrair_candidatos(img)` devolve os candidatos antes do filtro
final, com `area`, `circular`, `fill`, `bbox` e `hull`; o `anotar.py` usa essa
função diretamente para evitar lógica duplicada.

---

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `src/ref/embaubaHSVmask.py` | Script base original do professor (referência — **não modificar**) |
| `src/core/deteccao.py` | **Núcleo**: HSV/morfologia/hull, filtro `eh_embauba` e `detectar_embaubas` (bounds) |
| `src/core/execucao.py` | Processamento paralelo (`multiprocessing.Pool`) + checkpoint/retomada |
| `src/core/estatisticas.py` | Figuras estatísticas e geração do `relatorio.md` |
| `src/main.py` | Entrada principal: output completo sem path ou JSON por imagem/pasta |
| `src/anotar.py` | UI OpenCV para revisar rótulos e caixas faltantes da validação |
| `data/tiles/` | 992 tiles JPEG (não versionados — ver link de download) |
| `data/validacao/` | Conjunto rotulado de validação (ver `data/validacao/index.md`) |
| `data/output/` | Saídas geradas pelo pipeline (não versionado) |
| `index.md` | Estrutura obrigatória do relatório |
