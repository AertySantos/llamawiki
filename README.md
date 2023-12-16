# Fine-tuning da GPT LLAMA2 com a Wikipedia em Português no Supercomputador Atena

Aerty Santos, Eduardo Oliveira.

| AVAILABLE DOWNLOADS |
| :------------------: |
| [DATASETS](#datasets) |
| [VIDEOS](#videos) |

## Index
- [Descrição](#descrição)
- [Llama2](#Llama2)
- [Fine-tuning](#Fine-tuning)
- [Requisitos do Sistema](#Requisitos-do-Sistema)
- [Instruções para Replicar](#Instruções-para-Replicar)
- [Testes iniciais](#Testesiniciais)
- [Fine-tuning Vs RAG](#Fine-tuningVsRAG)
- [Perguntas e Respostas](#Perguntas-e-Respostas)
- [Vídeos](#Vídeos)
- [Referências](#Referências)

## Descrição
Poucas empresas tem dados e poder computacional necessário para treinar um grande modelo de linguagem, a maioria das empresas que conseguem guardam a sete chaves tais modelos. O simples fato de executar, em uma máquina de nível de consumidor, é impossível dada a exigência de muitos recursos.
Este artigo utilizará como base um grande modelo de linguagem GTP (Generative Pre-trained Transformer) disponibilizado de forma open-source pelo grupo Meta AI chamado Llama 2 e buscará investigar o processo de seu refinamento (fine-tuning), ou seja, uma personalização, utilizando os dados da Wikipedia em Português. A pesquisa foi conduzida utilizando a capacidade computacional do supercomputador Atena, que atualmente conta com 3 (três) GPUS A-100 de 80GB cada e mais de 500GB de memória RAM. Ao final do experimento será possível a comparação dos resultados de perguntas em três momentos: antes, sem nenhuma adição de informação; após o fine-tuning; e uma última comparação utilizando outra estrutura de recuperação de respostas, conhecida como RAG (Retrieval Augmented Generation). O objetivo central deste artigo é comparar as repostas e investigar a ampliação da capacidade de compreensão do modelo Pré-Treinado avaliando sua capacidade  geração de texto do modelo na língua portuguesa. 

## Llama2
Em julho de 2023, a Meta AI lançou o Llama 2 sendo disponibilizado em algumas versões segundo o seu tamanho e finalidade. Quando dito tamanho é a quantidade de tokens utilizados para o seu treinamento: 7, 13, 34 ou 70 bilhões de parâmetros, respectivamente conhecidos como modelos 7B, 13B, 34B  e 70B. Quanto maior a quantidade de parâmetros de entrada, mais o modelo aprende, podemos traçar uma comparação a uma criança que quanto mais ela vê e tem acesso a informação, mais ela aprende. Frequentemente as empresas utilizam os dados disponibilizados na internet, publicações digitalizadas, dados de usuários.. etc. De tal forma que o modelo 70B foi exposto 10x mais informações que o modelo 7B.  Quanto maior o número de parâmetros, mais o modelo aprende e aumenta sua precisão, no entanto, a exigência do poder computacional para seu treinamento e execução aumentam proporcionalmente, a medida que são expostos a quantidade de dados, o que pode se tornar um impecilho, pensando nisso, a META disponibilizou modelos menores e menos exigentes. Em resumo todos modelos do Llama2 trouxeram melhorias notáveis em relação à sua versão anterior Llama 1 disponibilizado em fevereiro de 2023, das quais se destacam:

-  Treinamento com 40% mais tokens, o que permite que os modelos aprendam mais informações sobre o mundo.
-  Uma extensão de contexto mais ampla (com até 4 mil tokens), o que permite que os modelos compreendam melhor conversas mais longas.
-  A implementação de atenção de consulta agrupada para uma inferência ágil nos modelos de 70B de parâmetros.
-  O destaque, no entanto, é a introdução dos modelos ajustados (Llama 2-Chat), otimizados para diálogos utilizando o Aprendizado por Reforço a partir do Feedback Humano (ARFH). O processo de ARFH é ilustrado na imagem abaixo.

![llama2](https://github.com/AertySantos/llamawiki/blob/master/llama2.png)

O destaque é que no ARFH, os modelos são treinados em um conjunto de dados de diálogos humanos. Em cada interação, o modelo gera uma resposta e recebe um feedback do humano. O feedback pode ser positivo (por exemplo, "Isso foi útil" ou "Isso foi seguro"), negativo (por exemplo, "Isso foi irrelevante" ou "Isso foi ofensivo") ou neutro (por exemplo, "Isso foi ok"). O modelo usa o feedback para melhorar suas respostas futuras. Ao longo do tempo, o modelo aprende a gerar respostas que são mais úteis, seguras e relevantes.

Em testes abrangentes de utilidade e segurança, os modelos Llama 2-Chat superaram muitos modelos abertos e atingiram um desempenho comparável ao ChatGPT 3.5 (privado), mesmo o GPT-3.5 tendo um tamanho de parâmetro de 175B. 

Os modelos Llama 2-Chat têm o potencial de melhorar significativamente a qualidade das interações entre humanos e máquinas. Eles podem ser usados em uma variedade de aplicações, incluindo chatbots, assistentes virtuais e sistemas de educação e se tornará ainda mais efetiva se conseguirmos adicionar conhecimento personalizado e adequados às nossas necessidades e orçamentos, o que nos leva ao cerne deste artigo, utilizando duas técnicas: Finetunnig e RAG.

## Fine-tuning
O ajuste fino no aprendizado de máquina é o processo de ajustar os pesos e parâmetros de um modelo pré-treinado em novos dados para melhorar seu desempenho em uma tarefa específica é uma prática frequentemente empregada para adaptar um LLM básico a um cenário de utilização específico. 
O potencial de economia de custos, a capacidade de processar dados confidenciais e até mesmo o potencial de desenvolver modelos que excedem o desempenho de modelos como ChatGPT e GPT-4 em determinadas tarefas específicas.
Embora não se exiga um poder computacional comparável ao necesário para geração do modelo, o ajuste fino pode ser um entrave. A própria META, informa em seu site ser possível ajustar o modelo 13B com uma única GPU de 24 GB de memória.
O treinamento é baseado em várias instruções, perguntas e respostas.

Os exemplos de treinamento costumam apresentar-se da seguinte maneira:
  
  \#\#\# Instruction:
 Analise a pergunta a seguir e responda de forma sucinta...
  
  \#\#\# Input:
 Qual a capital do Brasil?
  
  \#\#\# Response:
 Brasília

Instructions:
Input:
Response:


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

A Figura compara os requisitos de memória de diferentes métodos de ajuste fino. Como pode ser visto, o QLoRa requer menos memória do que o LoRA e o ajuste fino completo.

Por outro lado, o TRL (Transformador de Aprendizado por Reforço) é uma biblioteca utilizada para treinar modelos de linguagem utilizando o paradigma de aprendizado por reforço. Sua API de Treinamento para Ajuste Fino Supervisionado (SFT) facilita a criação de modelos personalizados e seu treinamento com conjuntos de dados customizados.

1. Execute o seguinte comando no terminal para instalar os pacotes Python necessários para o Fine-tuning:
   ```
   !pip install -q -U bitsandbytes
   !pip install -q -U git+https://github.com/huggingface/transformers.git
   !pip install -q -U git+https://github.com/huggingface/peft.git
   !pip install -q -U git+https://github.com/huggingface/accelerate.git
   ```
2. Execute o seguinte comando no terminal para executar o Fine Tune no Llama2-13b:
   ```
   python3 llama2-finetune.py
   ```
3. Digite o seguinte comando no terminal para testar o Fine Tune no Llama2-13b:
   ```
   python3 chat_fine.py
   ```

## Requisitos do Sistema
Preferencialmente, Sistema operacional: Ubuntu.  Pacotes: wget, md5sum.  Gerenciador de pacotes: Conda ME
Você deve ter o Python 3.9 ou posterior instalado. Versões anteriores do Python podem não compilar.
O Llama2 é opensource, no entanto, ele querer que você aceite os termos e solicite a licença e url pessoais para ter acesso aos modelos.

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



## RAG
Ao considerar a importância da avaliação das respostas geradas pelos Modelos de Linguagem de Aprendizado (LLMs), percebemos que são treinados com milhões de parâmetros, exigindo uma análise criteriosa para garantir a qualidade das conclusões. Nesse contexto, a Geração Aumentada de Recuperação (RAG) surge como uma abordagem que busca melhorar a qualidade das respostas do LLM, incorporando fontes externas de conhecimento. Este repositório explora como a RAG pode aprimorar a representação e a confiabilidade das respostas do LLM, considerando a sua estrutura e a integração de recursos externos durante o processo de geração.

![Processo do RAG](https://github.com/AertySantos/llamawiki/blob/master/rag.png)

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

3 - Qual era o nome do país antes de ser chamado de Brasil?
- Llama2-13b : <br>
A Ilha de Vera Cruz foi o nome dado pelos portugueses à região que hoje em dia é conhecida como o Brasil, quando foi descoberta em 1500 por Pedro Álvares Cabral. O nome "Brasil" só foi adotado oficialmente em 1822, após a independência do país.

- Llama2-13b RAG : <br>
Não é possível responder à pergunta com base nas informações fornecidas, pois o texto não menciona qualquer país ou território antes da formação do Brasil. O texto se concentra em descrições da vida no Nordeste do Brasil na época da independência e na Guerra do Paraguai.

- Llama2-13b fine tuning : <br>
O nome do país antes de ser chamado de Brasil era "Ilha de Vera Cruz" ou "Terra de Vera Cruz".
Fonte: "História do Brasil", de Lilia Moritz Schwarcz.

  A Ilha de Vera Cruz foi descoberta em 1500 por Pedro Álvares Cabral, que a batizou em homenagem à Virgem Santa Maria, padroeira de Portugal. A partir de então, o nome foi usado para se referir ao território que hoje é o Brasil.
A partir do século XVII, o nome "Brasil" passou a ser usado com mais frequência, mas o nome "Ilha de Vera Cruz" continuou a ser usado oficialmente até o século XIX.

4 - Quando foi proclamada a independência do Brasil?
- Llama2-13b : <br>
7 de setembro de 1822.
  
- Llama2-13b RAG : <br>
 A independência do Brasil foi proclamada em 1822.

- Llama2-13b fine tuning : <br>
A independência do Brasil foi proclamada em 7 de setembro de 1822. A declaração de independência foi feita por Dom Pedro I, que foi aclamado como imperador do Brasil. A independência do Brasil foi reconhecida oficialmente em 12 de outubro de 1822, quando o Tratado de Rio de Janeiro foi assinado entre o Brasil e Portugal.

5 - Quando foi assinada a primeira Constituição brasileira?
- Llama2-13b : <br>
A primeira Constituição brasileira foi assinada em 21 de março de 1824.

- Llama2-13b RAG : <br>
Não sou capaz de responder à sua pergunta, pois a constituição não existe. A Constituição Federal brasileira é uma lei que estabelece as normas fundamentais do país e foi promulgada em 1988. Antes disso, o Brasil era governado por um regime militar, e a última constituição anterior à atual foi promulgada em 1967.

- Llama2-13b fine tuning : <br>
A primeira Constituição brasileira foi assinada em 24 de fevereiro de 1824. Ela foi promulgada por Dom Pedro I, que havia sido proclamado imperador em 1822. A Constituição estabeleceu a monarquia constitucional no Brasil e estabeleceu os princípios da liberdade, da igualdade e da democracia.

6 - Quem foi a primeira presidente mulher do Brasil?
- Llama2-13b : <br>
Dilma Rousseff.

- Llama2-13b RAG : <br>
Não é possível responder à pergunta com base nas informações fornecidas, pois Fernando Collor de Mello é um homem e não uma mulher. A primeira presidente mulher do Brasil foi Dilma Rousseff, que governou de 2011 a 2016.

- Llama2-13b fine tuning : <br>
Dilma Rousseff foi a primeira presidente mulher do Brasil, eleita em 2010 e reeleita em 2014. Ela foi a 36° presidente do Brasil e governou até 2016, quando foi afastada do cargo por impeachment.

7 - Quem foi o primeiro imperador do Brasil?
- Llama2-13b : <br>
Dom Pedro I.

- Llama2-13b RAG : <br>
 The first emperor of Brazil was Pedro I (1822-1831).

- Llama2-13b fine tuning : <br>
Dom Pedro I (1822-1831) foi o primeiro imperador do Brasil. Ele foi coroado em 1 de dezembro de 1822, após a proclamação da independência do Brasil em 7 de setembro de 1822. Dom Pedro I foi o único imperador do Brasil, governando por cerca de nove anos antes de abdicar em favor de seu filho, Dom Pedro II.

8 - Qual foi o período do governo de Getúlio Vargas?
- Llama2-13b : <br>
O período do governo de Getúlio Vargas foi de 1930 a 1945, e novamente de 1951 a 1954. Ele governou o Brasil como ditador durante o regime conhecido como Estado Novo, e posteriormente como presidente eleito após a redemocratização do país.

- Llama2-13b RAG : <br>
O período do governo de Getúlio Vargas foi de 1930 a 1945, e novamente de 1951 a 1954.

- Llama2-13b fine tuning : <br>
O período do governo de Getúlio Vargas foi de 1930 a 1945, com interrupções. Ele foi eleito presidente em 1930 e governou até 1945, quando foi deposto. Durante seu governo, houve muitas mudanças políticas, econômicas e sociais no Brasil. Ele implementou políticas de contenção da inflação, estabilizou a moeda e implementou programas de desenvolvimento econômico. Ele também estabeleceu o Estado Novo, um regime autoritário que durou de 1937 a 1945.

9 - Quando foi abolida a escravidão no Brasil?
- Llama2-13b : <br>
A escravidão foi abolida no Brasil em 13 de maio de 1888, através da Lei Áurea.

- Llama2-13b RAG : <br>
A escravidão no Brasil foi abolida em 13 de maio de 1888, conforme a Lei Áurea (Lei nº 3.353).

- Llama2-13b fine tuning : <br>
A escravidão foi abolida no Brasil em 13 de maio de 1888, por meio da Lei Áurea, assinada por D. Pedro II. Essa lei aboliu a escravidão em todo o território brasileiro, libertando cerca de 20 milhões de escravos. No entanto, a escravidão continuou a existir de facto em alguns estados, especialmente no Nordeste, até o final do século XIX.

10 - Quem foi o presidente durante a Revolução de 1930?
- Llama2-13b : <br>
Getúlio Vargas.

- Llama2-13b RAG : <br>
Não posso responder à sua pergunta, pois não tenho conhecimento sobre a Revolução de 1930.

- Llama2-13b fine tuning : <br>
Getúlio Vargas foi o presidente durante a Revolução de 1930. Ele foi eleito em 1929 e tomou posse em 1º de março de 1930. A Revolução de 1930 ocorreu em outubro daquele ano, quando Getúlio Vargas foi deposto e substituído pelo marechal Pedro de Toledo.

11 - Quando foi fundada a cidade de Brasília?
- Llama2-13b : <br>
21 de abril de 1960.
  
- Llama2-13b RAG : <br>
 I don't know.

- Llama2-13b fine tuning : <br>
A cidade de Brasília foi fundada em 21 de abril de 1960.

12 - Qual foi o primeiro presidente eleito após a ditadura militar?
- Llama2-13b : <br>
O primeiro presidente eleito após a ditadura militar no Brasil foi Tancredo Neves. Ele foi eleito em 1984 e tomou posse em 15 de março de 1985, encerrando o período de ditadura militar que durou desde 1964.

- Llama2-13b RAG : <br>
 O primeiro presidente eleito após a ditadura militar foi Justiniano Borgoño.

- Llama2-13b fine tuning : <br>
Tancredo Neves foi o primeiro presidente eleito após a ditadura militar, em 1985. Ele venceu as eleições presidenciais com 48% dos votos, mas não pôde tomar posse devido à sua morte em 21 de abril de 1986. Em 25 de julho de 1986, o vice-presidente José Sarney foi empossado como presidente.

13 - Quem foi o primeiro presidente do período republicano?
- Llama2-13b : <br>
Deodoro da Fonseca.

- Llama2-13b RAG : <br>
O primeiro presidente do período republicano foi Deodoro da Fonseca.

- Llama2-13b fine tuning : <br>
Deodoro da Fonseca foi o primeiro presidente do Brasil durante o período republicano, tendo sido eleito em 15 de novembro de 1889 e exercendo o cargo de 15 de novembro de 1889 a 23 de março de 1891.

14 - Quais foram os objetivos e resultados da Campanha "O Petróleo é Nosso"?
- Llama2-13b : <br>
The "O Petróleo é Nosso" (The Oil is Ours) campaign was a significant event in the history of Brazil, aimed at raising awareness and promoting the country's oil industry. The campaign was launched in the 1970s by the military government, which ruled Brazil from 1964 to 1985.

- Llama2-13b RAG : <br>
The campaign "O Petróleo é Nosso" was a nationalist movement that aimed to promote the control of the Brazilian state over the country's oil resources. The movement emerged in 1948, during the government of President Getúlio Vargas, who had proposed a law to grant the state-owned oil company, Petrobras, the monopoly on oil exploration and production in Brazil. The campaign was led by the Centro de Estudos e Defesa do Petróleo (CEDPEN), an entity created by military officers, intellectuals, and students to promote the nationalization of the oil industry. The movement gained significant support among students, professionals, and military personnel, and it was marked by protests, rallies, and other forms of popular mobilization. Despite its efforts, the campaign ultimately failed to achieve its objectives, as President Vargas's government did not pass the

- Llama2-13b fine tuning : <br>
   A Campanha "O Petróleo é Nosso" foi uma das mais importantes da história do Brasil, pois teve como objetivo principal a nacionalização do petróleo, ou seja, a tomada de controle total do petróleo encontrado no país pela Petrobrás, empresa estatal que passou a ser responsável pela exploração e produção de petróleo no Brasil.
   
   A campanha foi liderada pelo presidente Juscelino Kubitschek, que era apoiado por uma grande maioria do Congresso Nacional, e teve como resultado a criação da Petrobrás em 1953. A empresa passou a ser responsável pela exploração e produção de petróleo no Brasil, e o país passou a ter controle total sobre suas reservas de petróleo.
   
   A campanha também teve como objetivo a redução dos preços do petróleo no Brasil, que eram muito altos devido ao monopólio da Standard Oil of New Jersey (atual ExxonMobil) e outras empresas estrangeiras que exploravam as reservas de petróleo no país. Com a nacionalização do petróleo, o governo brasileiro passou a ter controle sobre os preços do combustível, o que ajudou a reduzir os custos dos produtos petrolíferos no país.
   
   A Campanha "O Petróleo é Nosso" também teve como resultado a criação de uma empresa estatal de aviação, a Varig, que passou a ser responsável pela exploração e produção de aviões no Brasil. A Varig foi criada em 1953, juntamente com a Petrobrás, e passou a ser uma das principais empresas estatais do Brasil.
   
   Em resumo, a Campanha "O Petróleo é Nosso" foi um marco importante na história do Brasil, pois teve como objetivo principal a nacionalização do petróleo e a redução dos preços do combustível no país. O resultado da campanha foi a criação da Petrobrás.


15 - O que foi a Revolução Constitucionalista de 1932 e suas repercussões?
- Llama2-13b : <br>
   Revolução Constitucionalista de 1932 foi um movimento armado ocorrido no Brasil em 1932, com o objetivo de depor o presidente da República, Getúlio Vargas, e estabelecer uma ditadura militar. A revolução foi liderada por políticos e militares descontentes com o governo de Vargas, que havia sido eleito em 1930 e estava implementando políticas consideradas autoritárias e populistas.
   
   As repercussões da Revolução Constitucionalista de 1932 foram significativas para o Brasil. Apesar de não terem alcançado seu objetivo inicial, a revolução contribuiu para o desgaste do regime varguista e pavimentou o caminho para a Revolução de 1945, que estabeleceu uma ditadura militar no país.
   
   - Llama2-13b RAG : <br>
   A Revolução Constitucionalista de 1932 foi um movimento político-militar que ocorreu no estado de São Paulo, Brasil, em 1932. A revolta foi liderada pela Legião Revolucionária, um grupo de militares e civis que buscavam a reforma política e social do país. A revolução culminou com a assinatura do Pacto da Washington Luiz, que estabeleceu a convocação para a Assembleia Nacional Constituinte em 1934. As repercussões incluíram a repressão governamental contra os líderes da revolta e a consolidação do regime ditatorial de Getúlio Vargas.

- Llama2-13b fine tuning : <br>
   A Revolução Constitucionalista de 1932 foi um movimento armado que ocorreu no Brasil em 1932, com o objetivo de depor o presidente Getúlio Vargas e restaurar a República Velha. A revolta foi liderada por um grupo de militares e civis que se opunham à ditadura de Vargas, que haviam sido eleitos em 1930.
   
   A revolta foi reprimida pelo governo federal, que enviou tropas para combater os revoltosos. A repressão foi brutal, com muitas mortes e feridos entre os revoltosos e civis. A revolução foi derrotada, e Vargas manteve o poder.
   
   As repercussões da revolução foram significativas. A repressão do movimento levou a uma onda de perseguição aos oposicionistas, com muitos deles sendo presos, exilados ou assassinados. A censura também foi implementada, e a imprensa passou a ser controlada pelo governo.
   
   A derrota da revolução também teve impactos na política brasileira. Vargas passou a consolidar sua ditadura, e o país passou a ser governado por um regime autoritário que duraria até 1945. A revolução também marcou o início da Era Vargas, que seria caracterizada por uma série de medidas autoritárias e intervenções do Estado na economia.
   
   A Revolução Constitucionalista de 1932 foi um marco importante na história do Brasil, e teve repercussões significativas na política, na sociedade e na cultura do país. A derrota do movimento armado marcou o início de uma era autoritária no Brasil, que duraria por quatro décadas.

16 - Qual foi o papel dos bandeirantes na história do Brasil?
- Llama2-13b : <br>
   Os bandeirantes foram um grupo de aventureiros e colonos portugueses que se estabeleceram no interior do Brasil no século XVIII e XIX. Eles desempenharam um papel importante na história do Brasil, principalmente na expansão territorial do país e na criação de novas cidades e vilas.
   
   Os bandeirantes eram em sua maioria descendentes de portugueses e indígenas, e alguns deles também tinham ancestralidade africana. Eles se organizavam em grupos armados e partiam das cidades costeiras do Brasil em busca de riquezas, como ouro, prata e pedras preciosas. Durante suas expedições, os bandeirantes encontravam novos territórios e estabeleciam novas rotas comerciais.

- Llama2-13b RAG : <br>
The bandeirantes were a group of explorers and colonizers who played a significant role in the history of Brazil. They were responsible for the discovery and settlement of much of the country's interior, particularly in the states of São Paulo and Minas Gerais. The bandeirantes were also instrumental in the exploitation of the region's natural resources, such as gold and other minerals.

- Llama2-13b fine tuning : <br>
   Os bandeirantes foram caçadores de escravos africanos que se dedicavam à captura de indígenas e a exploração de recursos naturais, como madeira, pedras preciosas e metais. Eles também eram responsáveis por expulsar os franceses da região sul do Brasil.
   
   Os bandeirantes eram liderados por capitães-donatários, que eram responsáveis por explorar as terras e capturar os indígenas. Eles também eram responsáveis por expulsar os franceses da região sul do Brasil.
   
   A atividade dos bandeirantes foi fundamental para a colonização do Brasil, pois permitiu a exploração de recursos naturais e a criação de encomendas, que eram responsáveis por explorar as terras e capturar os indígenas.
   
   A história dos bandeirantes é controversa, pois muitos deles eram escravizados e explorados pelos capitães-donatários. Eles também eram responsáveis por expulsar os franceses da região sul do Brasil, o que levou a conflitos armados.
   
   Apesar de sua contribuição para a colonização do Brasil, a história dos bandeirantes é muitas vezes negligenciada ou esquecida na história oficial do país. Isso se deve ao fato de que muitos deles eram escravizados e explorados pelos capitães-donatários, o que levou a uma visão negativa da atividade dos bandeirantes.
   
   No entanto, a história dos bandeirantes é importante para entender a colonização do Brasil e a forma como o país se desenvolveu ao longo dos séculos. Eles foram fundamentalmente importantes para a exploração de recursos naturais e a criação de encomendas, que eram responsáveis por explorar as terras e capturar os indígenas.


17 - Quem foram os presidentes durante o período da Ditadura Militar no Brasil?
- Llama2-13b : <br>
   Durante o período da Ditadura Militar no Brasil, que durou de 1964 a 1985, houve quatro presidentes:
   
   1. Castelo Branco (1964-1967)
   2. Artur da Costa e Silva (1967-1969)
   3. Emílio Garrastazu Médici (1969-1974)
   4. Ernesto Geisel (1974-1985)
   
   Esses presidentes foram escolhidos e nomeados pelos militares, que exerciam o poder político e social no país durante a ditadura.

- Llama2-13b RAG : <br>
   Não é possível responder a essa pergunta com base nas informações fornecidas, pois a Ditadura Militar no Brasil ocorreu antes da existence dos presidentes que são conhecidos atualmente. A Ditadura Militar durou de 1964 a 1985 e os presidentes foram:
   
   * Marshal Artur da Costa e Silva (1964-1967)
   * General Humberto de Alencar Castelo Branco (1967-1970)
   * General Emílio Garrastazu Médici (1970-1974)
   * General Ernesto Geisel (1974-1979)
   * General João Figueiredo (1979-1985)

- Llama2-13b fine tuning : <br>
   Durante o período da Ditadura Militar no Brasil, houve quatro presidentes:
   
   1. Marshal Humberto de Alencar Castelo Branco (1964-1967)
   2. General Augusto Rademaker Pinto (1967-1969)
   3. General Emílio Garrastazu Médici (1969-1974)
   4. General Ernesto Geisel (1974-1979)
   
   Esses presidentes foram escolhidos pelo Conselho de Segurança Nacional (CSN) e exerceram poderes ditatoriais durante o período da Ditadura Militar.

18 - Quantos estados tem o Brasil?
- Llama2-13b : <br>
Resposta esperada: 26 estados.

- Llama2-13b RAG : <br>
O Brasil tem 26 estados.

- Llama2-13b fine tuning : <br>
    Brasil tem 26 estados.
   
   Explicação: O Brasil é um país federalista, dividido em 26 estados, que são:
   
   1. Acre
   2. Alagoas
   3. Amazonas
   4. Bahia
   5. Ceará
   6. Espírito Santo
   7. Goiás
   8. Maranhão
   9. Mato Grosso
   10. Mato Grosso do Sul
   11. Minas Gerais
   12. Pará
   13. Paraná
   14. Pernambuco
   15. Piauí
   16. Rio de Janeiro
   17. Rio Grande do Norte
   18. Roraima
   19. Santa Catarina
   20. São Paulo
   21. Sergipe
   22. Tocantins
   
   Cada estado tem sua própria constituição, governo e leis, mas todos são governados pelo presidente da República.

19 - Como a diversidade cultural impacta a identidade nacional do Brasil?
- Llama2-13b : <br>
   A diversidade cultural tem um impacto significativo na identidade nacional do Brasil. O país é conhecido por sua riqueza cultural, que é resultado da miscigenação de diferentes etnias e culturas. A presença de indígenas, africanos e europeus, entre outros, contribuiu para a formação da identidade brasileira.
   
   A diversidade cultural influenciou a forma como os brasileiros se veem e se relacionam uns com os outros. Ao longo da história, o Brasil tem sido um país de imigração, o que resultou na criação de uma sociedade multicultural. Essa diversidade é refletida na música, na culinária, na literatura e nas artes, entre outras áreas.
   
   A identidade nacional do Brasil também é influenciada pela diversidade cultural. Ao reconhecer e respeitar as diferenças culturais, os brasileiros são capazes de se sentir parte de uma comunidade mais ampla. A diversidade cultural ajuda a promover o respeito mútuo e a tolerância, o que é fundamental para a construção de uma sociedade mais inclusiva e democrática.
   
   No entanto, a diversidade cultural também pode gerar desafios. Ao longo da história, o Brasil tem enfrentado questões como discriminação e preconceito, que ainda persistem hoje em dia. É importante que sejam tomadas medidas para garantir o respeito e a igualdade para todos, independentemente de sua origem cultural.
   
   Em resumo, a diversidade cultural tem um impacto significativo na identidade nacional do Brasil. Ela contribui para a formação da identidade brasileira e ajuda a promover o respeito mútuo e a tolerância. No entanto, é importante que sejam tomadas medidas para garantir o respeito e a igualdade para todos, independentemente de sua origem cultural.

- Llama2-13b RAG : <br>
The diversity of cultures in Brazil has had a significant impact on the country's national identity. The country's history of colonization and immigration has resulted in a diverse population, with influences from indigenous peoples, Portuguese, African, and other cultures. This diversity has contributed to the richness of Brazilian culture, but it also poses challenges to the country's national identity. The coexistence of different cultures within the same nation can lead to conflicts and tensions, as well as the risk of cultural homogenization. However, this diversity can also be a source of strength for the country, fostering creativity, innovation, and resilience. Ultimately, the impact of cultural diversity on Brazil's national identity is complex and multifaceted, and it continues to evolve as the country navigates its place in the globalized world.

- Llama2-13b fine tuning : <br>
   A diversidade cultural do Brasil é um aspecto importante de sua identidade nacional. A presença de diferentes grupos étnicos, religiosos, linguísticos e culturais contribuiu para a formação da identidade brasileira. A diversidade cultural é uma das características mais importantes do Brasil e é uma das razões pelas quais o país é considerado um modelo de convivência pacífica entre diferentes grupos étnicos.<br>
   
   A diversidade cultural do Brasil é resultado da miscigenação entre os grupos étnicos que chegaram ao país a partir do século XV. A presença de africanos, europeus, indígenas e asiáticos contribuiu para a formação de uma identidade nacional única. A diversidade cultural também é refletida na culinária, na música, na literatura e em outras expressões culturais do país.<br>
   
   A diversidade cultural do Brasil é uma das razões pelas quais o país é considerado um modelo de convivência pacífica entre diferentes grupos étnicos. A presença de diferentes culturas e religiões no país contribuiu para a formação de uma identidade nacional que valoriza a diversidade e a convivência pacífica. A diversidade cultural também é uma das razões pelas quais o Brasil é considerado um país multicultural e multirreligioso.<br>
   
   No entanto, a diversidade cultural do Brasil também tem suas desvantagens. A presença de diferentes grupos étnicos e culturais pode gerar conflitos e desafios, especialmente quando há desigualdades econômicas e sociais. Além disso, a diversidade cultural pode ser uma fonte de preconceito e discriminação, especialmente contra os grupos étnicos e culturais minoritários.<br>
   
   Em resumo, a diversidade cultural do Brasil é uma das características mais importantes de sua identidade nacional. A presença de diferentes grupos étnicos, religiosos, linguísticos e culturais contribuiu para a formação de uma identidade nacional única e valoriza a diversidade e a convivência pacífica. No entanto, a diversidade cultural também tem suas desvantagens.

20 - ?
- Llama2-13b : <br>

- Llama2-13b RAG : <br>

- Llama2-13b fine tuning : <br>

## Datasets
[Dados do treino Fine tune](https://github.com/AertySantos/llamawiki/tree/master/json)
## Videos
## Referências
[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf)<br>
[QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)<br>

