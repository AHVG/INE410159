# CLAUDE.md

Leia o [AGENTS.md](AGENTS.md) para entender o projeto, estrutura de arquivos e regras.

## Notas específicas para Claude

- `src/ref/embaubaHSVmask.py` **nunca deve ser modificado nem importado** — é a referência original do professor (fica em `src/ref/` para não ser confundido com um ponto de entrada).
- Todo o código fica em `src/`; os dados e artefatos em `data/` (`tiles/`, `validacao/`, `output/`). Só `.md`, `run.sh` e `.gitignore` ficam na raiz.
- O núcleo fica em `src/core/` (`deteccao.py`, `execucao.py`, `estatisticas.py`) e usa imports relativos entre si. As únicas entradas públicas são `main.py` (pipeline completo sem path, JSON por imagem/pasta com path) e `anotar.py` (UI OpenCV de validação).
- Caminhos de dados nos pontos de entrada são resolvidos a partir de `__file__` (apontam para `../data`), então rodam de qualquer cwd.
- Priorizar simplicidade: sem abstrações desnecessárias, sem tratamento de erros para casos impossíveis.
- O pipeline paralelo usa `multiprocessing.Pool` — funções worker devem ser top-level (não aninhadas).
- `matplotlib.use('Agg')` deve vir antes de qualquer import de `matplotlib.pyplot` para funcionar em ambiente headless.
