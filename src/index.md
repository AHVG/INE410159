# Código-fonte

## Pontos de entrada

| Arquivo | Descrição |
|---------|-----------|
| `main.py` | Entrada principal. Sem argumentos: processa `data/tiles/` e gera artefatos completos em `data/output/`. Com caminho: retorna JSON por imagem e, opcionalmente, visualizações. |
| `anotar.py` | UI OpenCV de validação. Abre as entradas ainda não validadas de uma seleção (tiles/globs/pasta, `--amostra N`) e deixa marcar falsos positivos e embaúbas faltantes sobre a saída do algoritmo; persiste em `data/validacao/<tile>/` (existir = validado). |

### Uso rápido

```bash
python3 src/main.py                                        # pipeline completo
python3 src/main.py <imagem_ou_pasta>                      # detecção avulsa → pasta por imagem (JSON + figuras)
python3 src/anotar.py tile_0905                            # valida uma entrada
python3 src/anotar.py data/tiles --amostra 20              # valida 20 tiles sorteados
python3 src/anotar.py data/validacao --resumo             # métricas do conjunto validado
```

## Núcleo (`core/`)

| Módulo | Descrição |
|--------|-----------|
| `core/deteccao.py` | Pipeline de visão computacional clássica: Gaussian Blur → segmentação HSV → morfologia (close/open) → extração de contornos → classificação por área, circularidade e solidez. Exporta `detectar()`, `analisar()` e `salvar_passo_a_passo()`. |
| `core/execucao.py` | Processamento paralelo com `multiprocessing.Pool` e gerência de checkpoints. Grava estado parcial a cada 50 tiles para permitir retomada de runs interrompidas. |
| `core/estatisticas.py` | Geração de figuras estatísticas (histogramas, cobertura HSV, tempo/memória) e do `data/output/relatorio.md`. |
