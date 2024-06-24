# Automatic Classification Methods


## Goal(s)
- Procurar formas de automatizar tarefas, procurando abandonar “heurísticas manuais”:
    - Classificação
    - Previsão
    - Extração
    - (…)

### Extra (Hipotético):
- Tentar perceber se com isto podemos implementar uma "framework" para construção de "custom decision-trees" que possamos utilizar?
    - Composto por 'nodes'
    - Cada node tem um "decision algorithm" escolhido de entre X possíveis
    - Input (instance rm) -> output label/score/etc


## How & "Qual a utilidade disto?"

- Introduzir modelos de AI para automatizar tarefas SIMPLES!
    Avaliando:
    - Recursos c/ viabilidade operacional
    - Performance (*result-oriented*)

- Contemplar introdução de vector-search no RM via Elastic.


## Use Case for Analysis

**Solution**: Mdados
**Description**: Olhar para um doc/digitalização, e tentar associar a documentos *planeados*.

Isto vai implicar um conjunto de heurísticas a definir, produzindo uma decision-tree, a fim de associar um **documento provável**.


### Solução "Tradicional":
- Ir buscar os campos extraídos da digitalização:
    - Atribuir pesos arbitrários a estes campos;
- Utilizar um ou mais destes campos para fazer uma query e ir buscar possíveis documentos planeados que façam match com esta digitalização; 
- Fazer match dos campos extraídos com campos dos docs planeados que vieram como resultado na query:
    - Calcular um score para cada um destes resultados com base nas "matches" dos campos
- Compilar um "set de sugestões" de possíveis documentos planeados, ordenados pelo nosso score heurístico.
    - Abaixo de um certo treshold não consideramos os documentos.

*Muita mão na massa*^

Muita decisão "humana" arbitrária, quais os campos a fazer match, melhor forma de calcular o score, etcetc

### Solution Concept
-> Queremos convergir para simplificar e diminuir a escala/tamanho/complexidade destes processos - utilizando **modelos capazes de generalizar conhecimento/features e fazer previsões sobre os dados com confiança elevada.**

- Treinar um ou vários modelos que recebam os vários features/campos importantes de uma dada definição, para este classificar instâncias como X ou Y.
    - No caso da Mdados, olhar para digitalizações e tentar dizer se é Contrato ou Manual com um valor de confiança alto.


## Architectures

### RandomForestClassifier (uses specific/explicit features)
* Usage: Classification tasks on structured/tabular data.
* Pros: Robustness, feature importance insights, handles both numerical and categorical data.
* Cons: Computational and memory-intensive, lower interpretability compared to single trees.

![alt text](images/rfc.png)

### LSTM (uses text representations)
* Usage: Sequential data tasks requiring memory of long-term dependencies.
* Pros: Effective for time series forecasting, NLP, and other sequential data tasks.
* Cons: Computationally intensive, requires substantial data and tuning, complex architecture.

![alt text](images/lstm.png)

### Transformer (uses text representations)
* Usage: Tasks involving long-range dependencies and large datasets, especially in NLP.
* Pros: Captures long-range dependencies without vanishing gradient problem, highly effective for NLP, scalable.
* Cons: Requires significant computational resources, complex to implement and tune, less straightforward interpretability.

![alt text](images/transformer.webp)


## Conclusões/Balanço

- RFClassifier: 
    - muito lightweight
        - mt facilmente corre nas maquinas da cob
    - ideal para tarefas simples **com domínios fechados de labels**
        - fácil inclusão em pipelines
    - qualidade dos resultados entre o "aceitável" e o "ideal"
        - +-90%

- LSTM:
    - em teoria o melhor balanço entre resource-requirements e qualidade dos resultados
        - mas os testes/investigação feitas deram resultados péssimos e não consegui colocar a rede
        a convergir sem aumentar consideravelmente o tamanho (logo, mais recursos ainda)
    - qualidade dos resultados quase inexistente :) 
    - nao tem modelos publicamente disponiveis, uma vez que a motivaçao para os distribuir
    só surgiu mais tarde com os transformers devido à quantidade de recursos necessarios
    para os treinar

- Transformer (DistillBERT 63M):
    - em teoria tem a melhor qualidade dos resultados
        - +-90%
    - como aprende representaçoes textuais, pode aprender padroes "escondidos"
    que não são claros para o ser humano - mas que também podem ser de 
    difícil interpretração
    - muito resource heavy...
        - poderia ser dificil de correr nas maquinas mais pequenas
        - poderiamos ter que, se necessario, retreinar modelos destes
        à noite
        - ou ter uma maquina da cob computacionalmente dedicada ao treino destes modelos,
        e que automaticamente fazia sync dos ficheiros dos pesos com o que está nos servidores
        (solução péssima)

