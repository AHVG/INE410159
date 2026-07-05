# Validação — Detecção de Embaúba

Conjunto rotulado para comparar versões do detector de embaúba e alimentar
comparações futuras no relatório.

## Origem

As anotações vêm de 153 candidatos produzidos pelo pipeline base (HSV +
morfologia + `área > 10.000 px²`) em 18 tiles. Cada candidato foi rotulado como
`embauba` ou `lixo`. A pasta também aceita caixas em `faltantes`, que representam
embaúbas visíveis no tile mas não detectadas pelo pipeline base.

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
  "revisado": false,
  "deteccoes": [
    {
      "id": 222,
      "label": "embauba",
      "bbox": [860, 1177, 314, 350],
      "area": 69508.5,
      "circular": 0.3769,
      "fill": 0.4144
    }
  ],
  "faltantes": [
    {
      "bbox": [120, 340, 180, 210]
    }
  ]
}
```

- `bbox`: `[x, y, w, h]` em pixels do tile.
- `area`, `circular` e `fill`: features usadas para comparar regras.
- `faltantes`: caixas de embaúbas presentes no tile e não detectadas.
- `revisado`: `true` quando o tile foi revisado manualmente.

## Revisão

```bash
python3 src/anotar.py tile_0905
python3 src/anotar.py data/validacao/tile_0905/tile_0905.json
python3 src/anotar.py caminho/nova_imagem.jpg
python3 src/anotar.py data/validacao --pendentes
python3 src/anotar.py data/validacao --resumo
```

Quando o alvo é uma imagem que ainda não tem JSON, o `anotar.py` cria
`data/validacao/<nome_da_imagem>/`, copia a imagem para essa pasta, gera
automaticamente o JSON inicial com `core.deteccao.extrair_candidatos()` e salva o
overlay `_vis.png`. Os rótulos iniciais seguem a regra atual do detector
(`embauba` para candidatos que passam no filtro final, `lixo` para os demais),
mas devem ser revisados manualmente. Use `--refazer` para recriar esse JSON
inicial.

Controles:

- clique numa caixa: alterna `embauba` / `lixo`;
- arrastar em área vazia: desenha uma caixa em `faltantes`;
- clique direito numa caixa azul: remove uma faltante;
- `u`: desfaz a última faltante;
- `s`: salva, marca `revisado: true` e avança;
- `n`: pula o tile atual sem salvar;
- `q` ou `ESC`: encerra a sessão sem salvar o tile atual.

Para o relatório, revise todos os tiles com `--pendentes`. Quando todos estiverem
com `revisado: true`, os labels e as caixas em `faltantes` formam uma referência
consistente para comparações futuras.
