# CLAUDE.md

Leia o [AGENTS.md](AGENTS.md) para entender o projeto, estrutura de arquivos e regras.

## Notas específicas para Claude

- `embaubaHSVmask.py` **nunca deve ser modificado nem importado** — é a referência original do professor.
- `relatorio_embauba.py` é o único arquivo de desenvolvimento; mantê-lo auto-contido.
- Priorizar simplicidade: sem abstrações desnecessárias, sem tratamento de erros para casos impossíveis.
- O pipeline paralelo usa `multiprocessing.Pool` — funções worker devem ser top-level (não aninhadas).
- `matplotlib.use('Agg')` deve vir antes de qualquer import de `matplotlib.pyplot` para funcionar em ambiente headless.
