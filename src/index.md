# Código-fonte

## Pontos de entrada

| Arquivo | Descrição |
|---------|-----------|
| `main.py` | Entrada principal. Sem argumentos: processa `data/tiles/` e gera artefatos completos em `data/output/`. Com caminho: retorna JSON por imagem e, opcionalmente, visualizações. |
| `anotar.py` | UI OpenCV para revisão manual. Permite marcar falsos positivos e embaúbas faltantes sobre a saída do algoritmo; persiste as correções em `data/validacao/<tile>/`. |

### Uso rápido

```bash
python3 src/main.py                                        # pipeline completo
python3 src/main.py <imagem_ou_pasta>                      # detecção avulsa → JSON
python3 src/main.py <imagem_ou_pasta> --vis --passo-a-passo
python3 src/anotar.py tile_0905                            # abre janela de revisão
python3 src/anotar.py data/validacao --pendentes           # lista tiles pendentes
```

## Núcleo (`core/`)

| Módulo | Descrição |
|--------|-----------|
| `core/deteccao.py` | Pipeline de visão computacional clássica: Gaussian Blur → segmentação HSV → morfologia (close/open) → extração de contornos → classificação por área, circularidade e solidez. Exporta `detectar()`, `analisar()` e `salvar_passo_a_passo()`. |
| `core/execucao.py` | Processamento paralelo com `multiprocessing.Pool` e gerência de checkpoints. Grava estado parcial a cada 50 tiles para permitir retomada de runs interrompidas. |
| `core/estatisticas.py` | Geração de figuras estatísticas (histogramas, cobertura HSV, tempo/memória) e do `data/output/relatorio.md`. |

## Referência (`ref/`)

| Arquivo | Descrição |
|---------|-----------|
| `ref/embaubaHSVmask.py` | Script base original do professor. **Não modificar e não importar.** Serve apenas como referência do algoritmo de partida. |
