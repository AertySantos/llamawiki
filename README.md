# Fine-tuning da GPT LLMA2 com a Wikipedia em Português no Supercomputador Atena(em construção)

Aerty Santos, Eduardo Oliveira.

| AVAILABLE DOWNLOADS |
| :------------------: |
| [DATASETS](#datasets) |
| [VIDEOS](#videos) |

## Index
- [Requisitos do Sistema](#Requisitos-do-Sistema)
- [Instruções para Replicar](#Instruções-para-Replicar)
- [Descrição](#descrição)
- [Llama2](#Llama2)
- [Testes iniciais](#Testesiniciais)
- [Fine-tuning](#Fine-tuning)
- [Fine-tuning Vs RAG](#Fine-tuningVsRAG)
- [Perguntas e Respostas](#Perguntas-e-Respostas)
- [Vídeos](#Vídeos)
- [Referências](#Referências)

## Requisitos do Sistema
Você deve ter o Python 3.9 ou posterior instalado. Versões anteriores do Python podem não compilar.

## Instruções para Replicar

1. Clone este repositorio localmente.
   ```
   git clone https://github.com/AertySantos/llamawiki.git
   cd llamawiki
   ```
 
2. Crie um ambiente virtual com conda e ative-o. Primeiro, certifique-se de ter o conda instalado. Em seguida, execute o seguinte comando:
   ```
   conda create -n llms python=3.11 -y && source activate llms
   ```

3. Execute o seguinte comando no terminal para instalar os pacotes Python necessários:
   ```
   pip install -r requirements.txt
   ```
## Descrição
Este artigo investiga o processo de refinamento (fine-tuning) do modelo de linguagem GPT (Generative Pre-trained Transformer) Llama2, utilizando a Wikipedia em Português. A pesquisa foi conduzida utilizando a capacidade computacional do supercomputador Atena, permitindo a comparação dos resultados de perguntas antes e depois do fine-tuning e também com outra estrutura de recuperação de respostas, conhecida como RAG (Retrieval Augmented Generation). O objetivo central é ampliar significativamente a capacidade de compreensão e geração de texto do modelo na língua portuguesa.
## Llama2
A versão Llama 2 apresenta uma gama de Modelos de Linguagem de Grande Porte (LLMs) pré-treinados e ajustados, variando de 7B a 70B em parâmetros. Estes modelos trazem melhorias notáveis em relação à sua versão anterior, incluindo:

-  Um treinamento com 40% mais tokens, o que permite que os modelos aprendam mais informações sobre o mundo.
-  Uma extensão de contexto mais ampla (com até 4 mil tokens), o que permite que os modelos compreendam melhor conversas mais longas.
-  A implementação de atenção de consulta agrupada para uma inferência ágil nos modelos de 70B de parâmetros.
-  O destaque, no entanto, é a introdução dos modelos ajustados (Llama 2-Chat), otimizados para diálogos utilizando o Aprendizado por Reforço a partir do Feedback Humano (ARFH). O processo de ARFH é ilustrado na imagem abaixo.

![llama2](https://github.com/AertySantos/llamawiki/blob/master/llama2.png)

No ARFH, os modelos são treinados em um conjunto de dados de diálogos humanos. Em cada interação, o modelo gera uma resposta e recebe um feedback do humano. O feedback pode ser positivo (por exemplo, "Isso foi útil" ou "Isso foi seguro"), negativo (por exemplo, "Isso foi irrelevante" ou "Isso foi ofensivo") ou neutro (por exemplo, "Isso foi ok").

O modelo usa o feedback para melhorar suas respostas futuras. Ao longo do tempo, o modelo aprende a gerar respostas que são mais úteis, seguras e relevantes.

Em testes abrangentes de utilidade e segurança, os modelos Llama 2-Chat superaram muitos modelos abertos e atingiram um desempenho comparável ao ChatGPT. Este avanço destaca a eficácia do aprendizado com feedback humano para otimizar interações em modelos de linguagem.

Os modelos Llama 2-Chat têm o potencial de melhorar significativamente a qualidade das interações entre humanos e máquinas. Eles podem ser usados em uma variedade de aplicações, incluindo chatbots, assistentes virtuais e sistemas de educação.
## Testes iniciais
Foram realizados testes com os modelos Llama2 de tamanhos 7B, 13B e 70B. 
Os testes foram realizados primeiro em CPU e depois em GPU.
É necessario baixar o modelo para pasta models.

Os modelos Llama2 foram eficientes e escaláveis em CPU. No entanto, ainda havia espaço para melhorias no desempenho de tarefas de geração de texto.
1. Execute o seguinte comando no terminal para executar o Llama2 via Cpu:
   ```
   python3 chat_cpu.py
   ```
O desempenho dos modelos Llama2 foi significativamente melhorado em GPU. As tarefas de geração de texto foram executadas até 10 vezes mais rápido em GPU do que em CPU.

2. Execute o seguinte comando no terminal para executar o Llama2 via Gpu:
   ```
   python3 chat_gpu.py
   ```

## Fine-tuning
O ajuste fino de instruções é uma prática frequente empregada para adaptar um LLM básico a um cenário de utilização específico. Os exemplos de treinamento costumam apresentar-se da seguinte maneira:
  
  \#\#\# Instruction:
 Analise a pergunta a seguir e responda de forma sucinta...
  
  \#\#\# Input:
 Qual a capital do Brasil?
  
  \#\#\# Response:
 Brasília

  Contudo, ao criar um grupo de dados de treinamento adequado para ser facilmente utilizado com bibliotecas HF (Hugging Face), é aconselhável optar pelo formato JSONL. Uma estratégia direta para realizar essa tarefa é gerar um objeto JSON em cada linha, contendo apenas um campo de texto para cada exemplo. Um exemplo dessa estrutura seria algo similar a:
  
  { "text": "Abaixo está uma instrução ... ### Instruction: Analise a pergunta a ... ### Input: Qual a... ### Response: Brasília" },<br>
  { "text": "Abaixo está uma instrução ... ### Instruction: ..." }<br>
  
Para realizar o treinamento, foi criado um dataset com parte dos dados da Wikipédia do Brasil [dataset](https://github.com/AertySantos/llamawiki/blob/master/json/saida.json). Esse dataset foi dividido em dois conjuntos, treino e teste, por meio do algoritmo [particiona.py](https://github.com/AertySantos/llamawiki/blob/master/particiona.py).<br>

Para a fase de refinamento, utilizaremos algumas bibliotecas específicas da Hugging Face (HF):

- [transformers](https://huggingface.co/docs/transformers/index)
- [peft](https://huggingface.co/docs/peft/index)
- [trl](https://huggingface.co/docs/trl/index)

O PEFT é uma ferramenta que permite ajustar Modelos de Linguagem de forma eficaz, sem a necessidade de alterar todos os parâmetros do modelo. Essa abordagem é vantajosa por vários motivos, incluindo:

- Redução dos custos computacionais e de armazenamento: o ajuste fino completo de um grande modelo de linguagem pode ser muito caro, tanto em termos de tempo de treinamento quanto de espaço de armazenamento. O PEFT pode reduzir significativamente esses custos, permitindo que os modelos sejam ajustados em hardware de consumo e armazenados em dispositivos menores.
- Superação dos problemas de esquecimento catastrófico: o ajuste fino completo pode causar o esquecimento catastrófico, um fenômeno em que o modelo perde a capacidade de realizar tarefas que já havia aprendido. O PEFT pode ajudar a evitar esse problema, pois congela a maioria dos parâmetros do modelo, preservando seu conhecimento prévio.
- Melhor desempenho em regimes de poucos dados: o PEFT pode ser especialmente vantajoso em regimes de poucos dados, onde o ajuste fino completo pode não ser possível. O PEFT pode ajudar o modelo a aprender com os dados disponíveis, mesmo que sejam limitados.

Métodos de PEFT

Existem vários métodos de PEFT, cada um com suas próprias vantagens e desvantagens. Um dos métodos mais populares é o LoRA, que seleciona um subconjunto dos parâmetros do modelo com base em sua importância. O QLoRa é uma variante do LoRA que utiliza quantização de 4 bits para reduzir ainda mais os requisitos de memória.

![Figura 1: Comparação dos requisitos de memória de diferentes métodos de ajuste fino](https://github.com/AertySantos/llamawiki/blob/master/qlora.png)

A Figura 1 compara os requisitos de memória de diferentes métodos de ajuste fino. Como pode ser visto, o QLoRa requer menos memória do que o LoRA e o ajuste fino completo.

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

## Perguntas e Respostas

1 - Qual a capital do Brasil?

- Llama2-13b : <br>
A capital do Brasil é Brasília, localizada no Distrito Federal.

- Llama2-13b RAG : <br>
The capital of Brazil is Brasília.

- Llama2-13b fine tuning : <br>
A capital do Brasil é Brasília, localizada no Distrito Federal, no centro-oeste do país. A cidade foi planejada e construída a partir da década de 1950 como uma nova capital para substituir Rio de Janeiro, que era a antiga capital colonial do Brasil. A transferência da capital federal para Brasília ocorreu em 21 de abril de 1960.

2 - Quem foi Tiradentes e qual foi sua importância histórica?

- Llama2-13b : <br>
  Tiradentes (1743-1792) foi um líder político e militar brasileiro, considerado o "pai da independência do Brasil". Ele foi um dos principais organizadores da Inconfidência Mineira, uma conspiração que visava a independência do Brasil da Portugal.
  
  Tiradentes nasceu em São João del-Rei, no estado de Minas Gerais, e estudou na Universidade de Coimbra, em Portugal. Depois de retornar ao Brasil, ele se envolveu em movimentos políticos e militares, e foi preso e torturado pela coroa portuguesa por sua atividade política.
  
  A importância histórica de Tiradentes reside em seu papel como líder da Inconfidência Mineira, que foi uma das primeiras movimentações independentistas do Brasil. A conspiração foi descoberta e reprimida pela coroa portuguesa, mas ela inspirou outros movimentos independentistas no país.
  
  Tiradentes também é lembrado por sua bravura e dedicação à causa da independência. Ele foi torturado e executado em 1792, mas seu legado como líder e patriota brasileiro sobreviveu ao longo dos séculos. Sua vida e seus ideais são celebrados em muitas cidades do Brasil, e ele é considerado um herói nacional.
  
  A importância histórica de Tiradentes também se estende além das fronteiras brasileiras. Ele foi um pioneiro no movimento independentista latino-americano, que inspirou outras nações da região a luchar por sua independência. Sua história é um exemplo de como uma pessoa pode mudar a história de um país e influenciar o mundo em torno dela.

- Llama2-13b RAG : <br>
  Tiradentes é considerado um dos principais líderes da independência do Brasil. Ele foi um advogado, militar e político que lutou pela causa da independência do país. Ele foi capturado e executado pelas autoridades portuguesas em 1792, mas seu legado como um defensor da liberdade e da democracia é ainda hoje lembrado e celebrado no Brasil.

- Llama2-13b fine tuning : <br>
  Tiradentes foi um líder político, militar e escritor brasileiro, considerado o "pai da independência do Brasil". Ele foi um dos principais líderes da Inconfidencia Mineira, movimento que buscava a independencia do Brasil da Coroa Portuguesa.
  
  Tiradentes foi condenado à morte por enforcamento em 21 de abril de 1792, por sua participação na Inconfidencia Mineira. Seu corpo foi esquartejado e suas partes foram espalhadas pelo interior de Minas Gerais como forma de terror ao povo. No entanto, sua figura passou a ser lendária e ele foi lembrado como um herói da independência do Brasil.
  
  Em 1822, após a proclamação da independencia do Brasil, Tiradentes foi reabilitado e seu nome foi dado a diversas cidades, ruas e monumentos em homenagem a sua participação na luta pela independência. Atualmente, ele é considerado um dos principais heróis da história do Brasil.

## Datasets
[Dados do treino Fine tune](https://github.com/AertySantos/llamawiki/tree/master/json)
## Videos
## Referências

