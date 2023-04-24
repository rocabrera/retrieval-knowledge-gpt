# **Introdução**

Os modelos LLMs vem recebendo bastante atenção recentemente devido a seu ótimo desempenho em responder diversos tipos de questões diferentes. 

Esses modelos normalmente são pagos e não sabemos se podemos confiar nas respostas, uma vez que eles podem estar alucinando (inventando uma resposta).

Entretanto muitas vezes não precisamos de um modelo tão robusto para diversos tipos de perguntas. As vezes um modelo que tem bom desempenho no meu conjunto de dados ja é suficiente. Antigamente isso era feito por meio de ```finetunning```, mas hoje em dia com a chegada de LLMs se viu necessário o uso de outras técnicas, uma vez que esses modelos normalmente são muito grandes para serem executados em somente uma máquina. Em 2020, Patrick Lewis e sua equipe proporam uma técnica chamada ```Knowledge Retrieval``` que iremos explorar nesse trabalho com modelos atuais para tentar entender possíveis aplicações e limitações.

https://arxiv.org/abs/2005.11401

# **Objetivos**

Então utilizando Knowledge Retrieval quero entender se:

1. Se é possível melhorar as respsotas dos LLMs.
2. Se modelos open sources podem ter desempenho semelhantes a modelos pagos.
3. Se esse método pode ser utilizado como uma fonte de consulta para ajudar o operador a validar a resposta do modelo.

# **Metodologia**

Essa técnica consiste em:

1. Aplicar um modelo de embedding em um conjunto de texto de domínio específico que sera denominado ```knowledge base```.
2. Utilizar uma busca de similaridade entre a pergunta e a knowledge base para encontrar os textos mais similares.
3. Alimentar o PROMPT com o contexto retirado e usar como entrada para o modelo LLMs escolhido.

Nesse projeto em especifico, utilizamos como conjundo de dados mais de 20 mil abstracts de artigos de hemofilia, onde os passos foram:

1. Foi aplicado em cada artigo um modelo de embedding.

2. Quando o usuário faz uma pergunta, é feito uma busca de similaridade entre cada artigo e a pergunta e retorna-se os 4 artigos com maior score.

3. Esses artigos retornados vão para o contexto do PROMPT, o qual é usado como entrada para o modelo LLMs escolhido.

# **Resultados**

Foi possível demonstrar que as respostas dos LLMs melhoram significamanete utilizando Knowledge Retrieval, mas essa melhora não é suficiente para permitir trocar modelos pagos por modelos Open Sources.

Por último, percebeu-se que é possível retornar a fonte de dados que o modelo se baseou para gerar a resposta. Dessa forma, ajudando modelos de LLMs a se tornarem mais confiáveis e permitir uso deles em situações mais criticas como exemplo na área da saúde.
