# Validação — Detecção de Embaúba

Conjunto rotulado para comparar versões do detector de embaúba e alimentar
comparações futuras no relatório.

## Origem

As anotações vêm de 50 tiles selecionados para cobrir casos com muitas, poucas e
nenhuma detecção automática. Cada detecção do pipeline é revisada como embaúba
válida ou falso positivo. A pasta também aceita caixas em `faltantes`, que
representam embaúbas visíveis no tile mas não detectadas pelo pipeline base.

Estado atual:

- 50 imagens validadas.
- 207 detecções avaliadas.
- 179 verdadeiros positivos, 28 falsos positivos e 79 faltantes.
- Precisão 86,5%, recall 69,4% e F1 77,0%.

## Estrutura

```
data/validacao/
├── tile_XXXX/
│   ├── tile_XXXX.jpg
│   ├── tile_XXXX.json
│   └── tile_XXXX_vis.png
└── index.md
```

| Arquivo | Conteúdo |
|---------|----------|
| `tile_XXXX.jpg` | tile original usado na revisão |
| `tile_XXXX.json` | labels, caixas e estado de revisão |
| `tile_XXXX_vis.png` | overlay numerado: verde = embaúba, vermelho = lixo, azul = faltante |

## Esquema do JSON

```json
{
  "tile": "tile_0750.jpg",
  "embaubas": [
    { "bbox": [860, 1177, 314, 350], "area": 69508.5, "poligono": [[870, 1180]] }
  ],
  "n_candidatos": 21,
  "n_deteccoes": 1,
  "coverage_pct": 3.4,
  "tempo_s": 0.15,
  "memoria_mb": 42.0,
  "falsos_positivos": [],
  "faltantes": [
    { "bbox": [120, 340, 180, 210] }
  ]
}
```

- `embaubas`: saída do detector — `bbox` `[x, y, w, h]` em px, `area` (px²) e `poligono` (convex hull).
- `falsos_positivos`: índices de `embaubas` marcados como erro na validação.
- `faltantes`: caixas de embaúbas presentes no tile e não detectadas.
- `n_candidatos`, `coverage_pct`, `tempo_s`, `memoria_mb`: métricas de diagnóstico do `analisar`.
- **Não há flag de revisão**: a existência do JSON já marca a entrada como validada.

De `embaubas`/`falsos_positivos`/`faltantes` recupera-se TP/FP/FN:
`TP = len(embaubas) - len(falsos_positivos)`, `FP = len(falsos_positivos)`, `FN = len(faltantes)`.

## Validação

```bash
python3 src/anotar.py tile_0905                 # valida uma entrada
python3 src/anotar.py tile_0905 tile_0120       # várias de uma vez
python3 src/anotar.py "tile_01*"                # glob
python3 src/anotar.py data/tiles --amostra 20   # amostra aleatória de 20
python3 src/anotar.py data/validacao --resumo   # métricas do conjunto validado
python3 src/anotar.py tile_0905 --refazer       # reanota do zero (descarta)
python3 src/anotar.py data/validacao --incluir-validadas   # reabre as já feitas
```

A seleção pode ser tiles, caminhos de imagem, globs ou uma pasta. Para cada
entrada **ainda não validada**, o `anotar.py` roda o detector
(`core.deteccao.analisar()`) em memória e abre a janela com as detecções
desenhadas. Ao **salvar**, cria `data/validacao/<nome>/` com a imagem, o JSON e o
overlay `_vis.png` — é esse JSON que marca a entrada como validada.

Controles:

- clique numa detecção: alterna falso positivo (verde ↔ vermelho);
- arrastar em área vazia: desenha uma embaúba faltante (azul);
- clique direito numa caixa azul: remove uma faltante;
- `u`: desfaz a última faltante;
- `s`: salva e avança;
- `n`: pula o tile atual sem salvar;
- `q` ou `ESC`: encerra a sessão sem salvar o tile atual.

Entradas já validadas são puladas por padrão (`--refazer`/`--incluir-validadas`
para reabrir). O `--resumo` calcula precisão, recall e F1 sobre todo o conjunto
validado.
