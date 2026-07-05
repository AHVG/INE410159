# Estrutura da Apresentação Final (Combinada)

Slides Beamer que cobrem as **duas soluções** para detecção e segmentação de copas
de _Cecropia pachystachya_ sobre o mesmo ortomosaico do MataDIV. Espelham a
estrutura do relatório ([`../relatorio/index.md`](../relatorio/index.md)).

Fonte: [`apresentacao.tex`](apresentacao.tex). Compilar com
[`compilar_apresentacao.sh`](compilar_apresentacao.sh) ou `./dev.sh apresentacao`.

> A Parte II já inclui as contagens finais por parcela/composição e estatísticas
> de área. Permanecem pendentes apenas métricas de teste, curvas e figuras
> qualitativas da frente neural.

---

## Roteiro dos slides

```
Abertura
  1. Título (duas abordagens)
  2. Contexto do trabalho (duas frentes, mesmo ortomosaico)
  3. Problema biológico e objetivo
  4. Área de estudo e ortomosaico (comum às duas frentes)

PARTE I — Visão Computacional Clássica            [slide divisor]
  5. Recorte e amostras (992 tiles, validação)
  6. Motivação da pipeline clássica
  7. Visão geral do pipeline (passo a passo)
  8. Etapas do pipeline (9 etapas)
  9. Filtro final de copa (regra de decisão)
 10. Resultados gerais (tabela)
 11. Distribuições espaciais (histogramas)
 12. Cobertura de vegetação e Top 10
 13. Validação manual (TP/FP/FN, P/R/F1)
 14. Exemplos da validação (TP/FP/FN)
 15. Desempenho computacional (tempo e memória)
 16. Limitações da abordagem clássica

PARTE II — Redes Neurais (SAM 3 + YOLOv8)         [slide divisor]
 17. Motivação: SAM 3 e curadoria por exclusão
 18. Pipeline semi-automatizado em 6 etapas
 19. Etapa 1 — geração do dataset
 20. Etapas 2 e 3 — divisão espacial e correção (tabela do dataset)
 21. Etapa 4 — treinamento do YOLOv8x (otimização bayesiana)
 22. Etapas 5 e 6 — inferência e segmentação
 23. Métricas e resultados

SÍNTESE — Comparação e Conclusões                 [slide divisor]
 24. Comparação das duas abordagens (tabela)
 25. Conclusões e trabalhos futuros
```

---

## Convenções visuais

- Tema custom "Canopy/Leaf/Moss" (verde), definido no preâmbulo do `.tex`.
- Cada parte começa com um **slide divisor** (`\partdivider{Parte X}{Título}`),
  fundo verde e círculos decorativos.
- Figuras da Parte I vêm de `data/output/` (histogramas, passo a passo, top 10,
  tempo/memória).
- Figuras da Parte II, quando existirem, virão de
  [`../rede_neural/figuras/`](../rede_neural/figuras/) e
  [`../rede_neural/resultados/`](../rede_neural/resultados/).

## Pendências da Parte II

1. Adicionar métricas de teste quando disponíveis: Precisão, Recall, F1, mAP50 e
   mAP50-95.
2. Inserir curvas Precision-Recall, curvas de treino e importância dos
   hiperparâmetros.
3. Inserir figuras qualitativas de detecção/segmentação como novos slides após o
   21.
