# Estrutura da Apresentação Final (Combinada)

Slides Beamer que cobrem as **duas soluções** para detecção e segmentação de copas
de _Cecropia pachystachya_ sobre o mesmo ortomosaico do MataDIV. Espelham a
estrutura do relatório ([`../relatorio/index.md`](../relatorio/index.md)).

Fonte: [`apresentacao.tex`](apresentacao.tex). Compilar com
[`compilar_apresentacao.sh`](compilar_apresentacao.sh) ou `./dev.sh apresentacao`.

> A Parte II usa `\textcolor{red}{[Pendente]}` onde faltam resultados da frente
> neural. Material-fonte em [`../rede_neural/`](../rede_neural/).

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
 14. Desempenho computacional (tempo e memória)
 15. Limitações da abordagem clássica

PARTE II — Redes Neurais (SAM 3 + YOLOv8)         [slide divisor]
 16. Motivação: SAM 3 e curadoria por exclusão
 17. Pipeline semi-automatizado em 6 etapas
 18. Etapa 1 — geração do dataset
 19. Etapas 2 e 3 — divisão espacial e correção (tabela do dataset)
 20. Etapa 4 — treinamento do YOLOv8x (otimização bayesiana)
 21. Etapas 5 e 6 — inferência e segmentação
 22. Métricas e resultados (PENDENTE)

SÍNTESE — Comparação e Conclusões                 [slide divisor]
 23. Comparação das duas abordagens (tabela)
 24. Conclusões e trabalhos futuros
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

## Ao preencher a Parte II

1. Adicionar as métricas reais no slide 22 e na tabela do slide 23 (célula
   "Resultado" → substituir `a definir`).
2. Inserir figuras qualitativas de detecção/segmentação como novos slides após o 21.
3. Remover os marcadores `\textcolor{red}{[Pendente]}`.
