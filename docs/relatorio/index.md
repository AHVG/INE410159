# Estrutura Obrigatória do Relatório

## 1. Introdução

- Contexto do problema: detecção de árvores Cecropia (Embaúba) em mosaico aéreo
- Objetivo do trabalho
- Abordagem escolhida (visão computacional clássica, sem deep learning)
- Justificativa da abordagem

## 2. Dataset

- Número de tiles e dimensões (ex: 992 tiles, 2048×2048 px)
- Sistema de coordenadas e resolução espacial
- Tamanho aproximado em disco
- Descrição de casos especiais (ex: tiles completamente pretos, fora da área mapeada)
- Descrição do conjunto `data/validacao/`: organização por pasta, JSONs de
  revisão, saída do detector em `embaubas`, índices `falsos_positivos` e caixas
  `faltantes`

## 3. Pipeline

Antes de detalhar as etapas, incluir uma subseção de motivação geral:

- **Motivação da pipeline**: explicar por que o problema foi dividido em
  etapas clássicas de visão computacional, como a sequência cor → morfologia →
  contornos → filtros geométricos ajuda na detecção de embaúba e quais
  limitações essa escolha procura controlar.

Depois da motivação geral, descrever **cada etapa** em subseção própria:

```md
### 3.X Nome da etapa

![Legenda curta da etapa](caminho/para/imagem.png)

**Motivação:** por que essa etapa foi escolhida — qual problema ela resolve no
contexto da detecção de embaúba.

**O que faz:** descrição objetiva da operação.

**Parâmetros:** valores utilizados e como foram determinados (empiricamente,
literatura, etc.).
```

Cada etapa deve ter uma imagem para ilustrar visualmente a transformação ou o
resultado intermediário correspondente. Preferir imagens do passo a passo
geradas pelo próprio pipeline em `data/output/passo_a_passo/`, quando
disponíveis.

### Etapas obrigatórias a cobrir

| Etapa | Operação |
|-------|----------|
| 1 | Suavização (Gaussian Blur) |
| 2 | Conversão de espaço de cor (BGR → HSV) |
| 3 | Limiarização por cor (faixa HSV das copas) |
| 4 | Fechamento morfológico (conecta fragmentos) |
| 5 | Abertura morfológica (remove ruído) |
| 6 | Detecção de contornos |
| 7 | Filtragem por área mínima |
| 8 | Filtro final de embaúba (área/circularidade) |
| 9 | Convex Hull (aproximação da copa) |

Observação: o Convex Hull também pode ser usado internamente na medição dos
candidatos (bbox, solidez e preenchimento), mas deve aparecer no relatório como
etapa final de regularização, visualização e exportação dos limites detectados.

## 4. Resultados

Usar a seguinte ordem de subseções:

### 4.1 Estatísticas gerais

- Total de tiles processados e tiles pretos ignorados
- Percentual de tiles com detecção
- Número total de detecções e área total estimada
- Tempo de processamento (total, médio por tile, mínimo e máximo, quando disponível)
- Estatísticas de detecções por tile (média, desvio padrão, máximo)
- Estatísticas de área por região detectada (média, mediana, máxima)
- Uso de memória médio e pico, quando disponível

### 4.2 Distribuição de detecções por tile

- Histograma de detecções por tile
- Breve interpretação da distribuição

### 4.3 Distribuição das áreas de detecção

- Histograma das áreas das regiões detectadas
- Breve interpretação da distribuição e da presença de regiões grandes

### 4.4 Cobertura HSV por tile

- Figura de cobertura HSV
- Explicar que cobertura HSV mede pixels dentro da faixa de cor após morfologia
  e não equivale diretamente a número de copas

### 4.5 Top 10 tiles com mais detecções

- Figura e/ou tabela com arquivo, detecções, área e cobertura

### 4.6 Validação manual

- Quantidade de tiles revisados
- Verdadeiros positivos (TP), falsos positivos (FP) e falsos negativos (FN)
- Precisão, recall e F1
- Breve interpretação do que precisão e recall indicam no contexto do método

### 4.7 Desempenho computacional

- Figuras de tempo e memória
- Comentário sobre custo por tile e viabilidade do pipeline clássico

## 5. Interpretação e Limitações

- Interpretação da taxa de tiles com detecções
- Discussão sobre falsos positivos por confusão espectral
- Discussão sobre regiões grandes/fundidas e limites dos filtros geométricos
- Leitura da validação manual: o que a precisão e o recall indicam

## 6. Conclusões

- Síntese dos resultados obtidos
- Limitações do método (falsos positivos, dependência de iluminação, sobreposição de tiles, etc.)
- Possíveis melhorias ou trabalhos futuros

---

## Arquivos desta pasta

| Arquivo | Descrição |
|---------|-----------|
| `relatorio.tex` | Fonte LaTeX do relatório escrito |
| `relatorio.pdf` | PDF compilado do relatório |
| `apresentacao.tex` | Fonte LaTeX dos slides (Beamer) |
| `apresentacao.pdf` | PDF compilado da apresentação |
| `compilar.sh` | Compila `relatorio.tex` → `relatorio.pdf` |
| `compilar_apresentacao.sh` | Compila `apresentacao.tex` → `apresentacao.pdf` |

```bash
cd relatorio/
./compilar.sh              # gera relatorio.pdf
./compilar_apresentacao.sh # gera apresentacao.pdf
```
