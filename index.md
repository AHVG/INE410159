# INE5443 — Detecção de Embaúba

Detecção de copas de **Cecropia pachystachya (Embaúba)** em ortomosaico de VANT usando visão computacional clássica (sem deep learning). Projeto final do Grupo 10 — UFSC.

## Estrutura do repositório

| Caminho | Descrição |
|---------|-----------|
| `src/` | Código-fonte — ver [`src/index.md`](src/index.md) |
| `docs/relatorio/` | Relatório escrito em LaTeX — ver [`docs/relatorio/index.md`](docs/relatorio/index.md) |
| `docs/apresentacao/` | Slides em LaTeX (Beamer) |
| `data/tiles/` | 992 tiles JPEG 2048×2048 px (não versionados) |
| `data/validacao/` | Conjunto rotulado para validação — ver [`data/validacao/index.md`](data/validacao/index.md) |
| `data/output/` | Artefatos gerados pelo pipeline (não versionados) |
| `dev.sh` | Script único: compila docs ou roda o projeto |
| `AGENTS.md` | Guia completo para agentes de IA |
| `CLAUDE.md` | Instruções específicas para Claude Code |

## Como executar

```bash
./dev.sh pipeline                            # pipeline completo → data/output/
./dev.sh anotar tile_0905                    # UI de revisão de validação
./dev.sh relatorio                           # compila docs/relatorio/relatorio.pdf
./dev.sh apresentacao                        # compila docs/apresentacao/apresentacao.pdf

python3 src/main.py <imagem_ou_pasta>        # detecção avulsa → JSON
python3 src/main.py <imagem_ou_pasta> --vis --passo-a-passo
```

Requer: `opencv-python`, `matplotlib`, `numpy`.

## Pipeline resumido

| Etapa | Operação | Parâmetros |
|-------|----------|------------|
| 1 | Gaussian Blur | kernel 9×9 |
| 2 | Segmentação HSV | H:[44,58] S:[144,231] V:[104,203] |
| 3 | Morfologia CLOSE | elipse 25×25 |
| 4 | Morfologia OPEN | elipse 9×9 |
| 5 | Filtro de contornos | área > 10.000 px² |
| 6 | Filtro embaúba | área ≥ 33.798 ou circularidade ≤ 0.1551 |
| 7 | Convex Hull | contorno final da copa |
