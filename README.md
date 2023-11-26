# Fine-tuning da GPT LLMA2 com a Wikipedia em Português no Supercomputador Atena(em construção)

Aerty Santos, Eduardo Oliveira.

| AVAILABLE DOWNLOADS |
| :------------------: |
| [DATASETS](#datasets) |
| [VIDEOS](#videos) |

## Index
<!-- Table of contents generated by http://tableofcontent.eu/ -->
- [Description](#description)
- [Llama2](#Llama2)
- [Fine-tuning](#Fine-tuning)
- [Fine-tuning Vs RAG](#Fine-tuningVsRAG)
- [Pré-requisitos](#Pré-requisitos)
- [Testes](#Testes)
## Description
Este artigo investiga o processo de refinamento (fine-tuning) do modelo de linguagem GPT (Generative Pre-trained Transformer) Llama2, utilizando a Wikipedia em Português. A pesquisa foi conduzida utilizando a capacidade computacional do supercomputador Atena, permitindo a comparação dos resultados de perguntas antes e depois do fine-tuning e também com outra estrutura de recuperação de respostas, conhecida como RAG (Retrieval Augmented Generation). O objetivo central é ampliar significativamente a capacidade de compreensão e geração de texto do modelo na língua portuguesa.
## Llama2
A versão Llama 2 apresenta uma gama de Modelos de Linguagem de Grande Porte (LLMs) pré-treinados e ajustados, variando de 7B a 70B em parâmetros. Estes modelos trazem melhorias notáveis em relação à sua versão anterior, incluindo um treinamento com 40% mais tokens, uma extensão de contexto mais ampla (com até 4 mil tokens) e a implementação de atenção de consulta agrupada para uma inferência ágil nos modelos de 70B de parâmetros. O destaque, no entanto, é a introdução dos modelos ajustados (Llama 2-Chat), otimizados para diálogos utilizando o Aprendizado por Reforço a partir do Feedback Humano (ARFH). Em testes abrangentes de utilidade e segurança, esses modelos superam muitos modelos abertos e atingem um desempenho comparável ao ChatGPT, conforme avaliações humanas. Este avanço destaca a eficácia do aprendizado com feedback humano para otimizar interações em modelos de linguagem.
## Fine-tuning
O ajuste fino de instruções é uma prática frequente empregada para adaptar um LLM básico a um cenário de utilização específico. Os exemplos de treinamento costumam apresentar-se da seguinte maneira:
  
  \#\#\# Instruction:
 Analise o comentário a seguir e classifique o tom como...
  
  \#\#\# Input:
  Eu amo ler seus artigos...
  
  \#\#\# Response:
  amigável e construtivo

  Contudo, ao criar um grupo de dados de treinamento adequado para ser facilmente utilizado com bibliotecas HF (Hugging Face), é aconselhável optar pelo formato JSONL. Uma estratégia direta para realizar essa tarefa é gerar um objeto JSON em cada linha, contendo apenas um campo de texto para cada exemplo. Um exemplo dessa estrutura seria algo similar a:
  
  { "text": "Abaixo está uma instrução ... ### Instruction: Analise o... ### Input: Eu amo... ### Response: amigável" },<br>
  { "text": "Abaixo está uma instrução ... ### Instruction: ..." }

Para a fase de refinamento, utilizaremos algumas bibliotecas específicas da Hugging Face (HF):

- [transformers](https://huggingface.co/docs/transformers/index)
- [peft](https://huggingface.co/docs/peft/index)
- [trl](https://huggingface.co/docs/trl/index)

O PEFT (Ajuste Fino Eficiente de Parâmetros) é uma ferramenta que permite ajustar os Modelos de Linguagem de forma eficaz, sem a necessidade de alterar todos os parâmetros do modelo. Esta ferramenta suporta o método QLoRa, possibilitando o ajuste de uma pequena porção dos parâmetros do modelo com quantização de 4 bits.

Por outro lado, o TRL (Transformador de Aprendizado por Reforço) é uma biblioteca utilizada para treinar modelos de linguagem utilizando o paradigma de aprendizado por reforço. Sua API de Treinamento para Ajuste Fino Supervisionado (SFT) facilita a criação de modelos personalizados e seu treinamento com conjuntos de dados customizados

## RAG
Ao considerar a importância da avaliação das respostas geradas pelos Modelos de Linguagem de Aprendizado (LLMs), percebemos que são treinados com milhões de parâmetros, exigindo uma análise criteriosa para garantir a qualidade das conclusões. Nesse contexto, a Geração Aumentada de Recuperação (RAG) surge como uma abordagem que busca melhorar a qualidade das respostas do LLM, incorporando fontes externas de conhecimento. Este repositório explora como a RAG pode aprimorar a representação e a confiabilidade das respostas do LLM, considerando a sua estrutura e a integração de recursos externos durante o processo de geração.

Por meio do Langchain, um framework reconhecido por simplificar a criação eficiente de aplicativos baseados em Modelos de Linguagem (LLM) e sistemas conversacionais, será viabilizado o carregamento do XML da Wikipedia em português. Dessa forma, entender a estrutura do Langchain torna-se fundamental para adotar uma abordagem unificada na criação e implementação padronizada de LLMs em diferentes aplicativos. Destaca-se a colaboração do Langchain com o Hugging Face - uma plataforma no GitHub que disponibiliza mais de 120 mil modelos - proporcionando um potencial significativo para o desenvolvimento de LLMs adaptáveis a uma ampla variedade de casos de uso, ampliando sua aplicabilidade e eficácia.

## Fine-tuning Vs RAG
A utilização de Modelos de Linguagem de Ajuste Fino (LLMs) resulta em sistemas adaptáveis capazes de lidar com uma vasta gama de tarefas em Processamento de Linguagem Natural (PNL). Estes modelos ajustados são especialmente eficazes em atividades como classificação de texto, análise de sentimento, geração de texto, e outras, centradas na compreensão e produção de texto a partir de entradas variadas.

Por outro lado, os modelos de Geração Aumentada de Recuperação (RAG) destacam-se em cenários onde a tarefa demanda o acesso a fontes externas de conhecimento. Esses modelos são especialmente relevantes para responder a perguntas em domínios abertos, resumir documentos extensos ou mesmo em chatbots capazes de oferecer informações provenientes de bases de conhecimento.

Sobre os Dados de Treinamento:

Os dados de treinamento para Modelos de Linguagem de Ajuste Fino (LLMs) são específicos da tarefa em questão, geralmente constituídos por exemplos rotulados correspondentes à tarefa almejada. No entanto, estes conjuntos de dados não incorporam diretamente mecanismos explícitos de recuperação.

Já os modelos de Geração Aumentada de Recuperação (RAG) são treinados para operações combinadas de recuperação e geração, normalmente utilizando uma mistura de dados supervisionados (para geração de conteúdo) e dados que demonstram como recuperar e efetivamente utilizar informações externas.

## Pré-requisitos

### Testes
Qual a capital do Brasil?
- Llama2-13b : <br>
A capital do Brasil é Brasília, localizada no Distrito Federal.
- Llama2-13b RAG : <br>
The capital of Brazil is Brasília.
- Llama2-13b fine tuning : <br>
A capital do Brasil é Brasília, localizada no Distrito Federal, no centro-oeste do país.

