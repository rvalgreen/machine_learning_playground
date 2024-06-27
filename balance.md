# Automatic Classification Viability

## Use Case for Analysis

**Solution**: Mdados
**Description**: Olhar para um digitalização, e tentar associar a documentos *planeados*.

- Uma digitalização pode corresponder a um Contrato, Requisição ou (Doc) Manual.

- Atualmente as digitalizações são enviadas para o DocDigitizer, de onde se extrai alguns campos.

- Depois alguém tem que manualmente olhar para elas e tentar associar a Requisição, Contrato ou (Doc) Manual:
    - Gasto de tempo considerável;
    - Vários campos a ter em conta para fazer o "matching";
    
## Goals
- Procurar formas de automatizar tarefas, procurando abandonar “heurísticas manuais”:
    - Classificação
    - Previsão
    - Extração
    - (…)

- Fazendo um levantamento/balanço de Custo x Benefício deste tipo de tecnologias.

## How

- Introduzir modelos de AI para automatizar tarefas SIMPLES!
    É preciso contemplar:
    - Recursos c/ viabilidade operacional
    - Performance (*result-oriented*)

- Contemplar introdução de vector-search no RM via Elastic (final)


### Exemplo de uma Solução "Tradicional":
1. Ir buscar os campos extraídos da digitalização:
    - Atribuir pesos arbitrários a estes campos;
2. Utilizar um ou mais destes campos para fazer uma query e ir buscar possíveis documentos planeados que façam match com esta digitalização; 
3. Fazer match dos campos extraídos com campos dos docs planeados que vieram como resultado na query:
    - Calcular um score para cada um destes resultados com base nas "matches" dos campos
4. Compilar um "set de sugestões" de possíveis documentos planeados, ordenados pelo nosso score calculado.
    - Abaixo de um certo treshold não consideramos os documentos.

#### Provavelmente dá bons resultados OU resultados bons o suficientes para "sugestões automáticas", mas envolve muita decisão "humana arbitrária" -> quais os campos a fazer match, melhor forma de calcular o score, etcetc -> coisas cuja responsabilidade pode ser passada para um modelo que é capaz de olhar para os vários features de uma definição e abstrair padrões relevantes e quantificar importância dos features  , enquanto produz resultados de elevada confiança.

### Novel Solution Concept
Ideia: Tentar alterar o processo da solução "tradicional", e introduzir uma pipeline com modelos pré-treinados nos dados disponíveis, a tentar diminuir:
- Escala
- Complexidade

Para este use case específico, o objetivo final é tentar chegar a uma das seguintes "conclusões":
- Temos um doc planeado sugerido, com um grau de confiança aceitável
- Temos um conjunto de docs planeados sugeridos, com um grau de confiança aceitável;
- Não conseguimos sugerir nenhum doc planeado porque não encontrámos ou as nossas previsões/sugestões não têm confiança aceitável.


### Como é que isto funcionaria?
Dependendo da tarefa que queremos cumprir (classificação, previsão, etc), temos que treinar um modelo pequeno e adequado.

Estes modelos vão olhar para os diversos features (valores dos campos em termos de RM) de uma dada definição, para tentarem classificar instâncias como X ou Y. Por exemplo:
    - Neste use case, vai olhar para digitalizações e tentar dizer se é Contrato ou Manual com um valor de confiança alto.

![alt text](<images/pipeline draft.png>)

#### Para isto funcionar, os classificadores em si têm que ser viáveis....

## Architectures

### RandomForestClassifier (uses specific/explicit features)
* Usage: Classification tasks on structured/tabular data.
* Pros: Robustness, feature importance insights.
* Cons: lower interpretability compared to single trees.

![alt text](images/rfc.png)

### LSTM (uses text representations)
* Usage: Sequential data tasks requiring memory of long-term dependencies.
* Pros: Effective for time series forecasting, NLP, and other sequential data tasks.
* Cons: Computationally intensive, requires substantial data and tuning, complex architecture.

![alt text](images/lstm.png)

### Transformer (uses text representations)
* Usage: Tasks involving long-range dependencies and large datasets, especially in NLP.
* Pros: Captures long-range dependencies without vanishing gradient problem, highly effective for NLP, scalable.
* Cons: Requires significant computational resources, complex to tune, less straightforward interpretability.

![alt text](images/transformer.webp)


## "Training Recipe"
Requirements: Python and some libraries :)

1. Load data
2. Pre-process data
    - Choose features/columns/fields
    - Choose textual representation (when using LSTM or Transformer)
3. Prepare 'Labels' column
4. Train model :)
5. Evaluate :)
6. Save model to disk :)

## Resumo dos Modelos

- RFClassifier: 
    - Muito lightweight / Resource friendly
        - apenas alguns segundos para "treinar" com milhares de samples
    - Ideal para tarefas simples **com domínios fechados**
        - Fácil inclusão em pipelines
    - Qualidade dos resultados entre o "aceitável" e o "ideal"
        - +-90%

- LSTM:
    - **Em teoria** o melhor balanço entre *resource-requirements* e qualidade dos resultados
        - Infelizmente os testes/investigação feitas deram resultados péssimos e não consegui colocar a rede
        a convergir sem aumentar consideravelmente o tamanho (logo, mais recursos ainda)
    - Não tem modelos publicamente disponiveis, uma vez que a motivaçao para os distribuir
    só surgiu mais tarde com os transformers devido à quantidade de recursos necessarios
    para os treinar.

- Transformer (DistillBERT 63M):
    - **Em teoria** tem a melhor qualidade de todos.
        - +-90%
    - Como aprende representações textuais, pode aprender padrões "escondidos"
    que não são claros para o ser humano - mas que também podem ser de 
    difícil interpretação...
    - O mais resource-hungry de longe...
        - À vontade demora >10min a treinar (em CPU)...
            - Mais lento nas maquinas mais pequenas
        - Poderíamos ter que, se necessário, retreinar modelos destes
        à noite....
        - Ou ter uma máquina dedicada ao treino destes modelos,
        e que automaticamente fazia sync dos ficheiros dos pesos com o que está nos servidores...
        (solução péssima)


## A contemplar:

Introduzir vector-search no RM via ES.

Como se implementaria isto:
1. Adicionar aos índices das definições que queremos um field (no ES) do tipo dense_vector
2. Ir a todas as instâncias existentes, representá-las sob a forma de uma String (em texto), e passá-las por um modelo DE EMBEDDINGS, para indexar a sua representação (em embeddings).
3. Fazer o mesmo para qualquer nova instância que seja adicionada (possível overhead)
4. Para uma query (linguagem natural) ou qualquer outra instância (representada sob a forma de texto), é transformada em embeddings, e utilizamos o algoritmo de k-NN search (que já é suportado default pelo ES) para obter os k resultados mais semelhantes de um dado índice.

-> Assim passava-se a suportar formas de calcular scores de semelhança entre query-instâncias ou instância-instâncias.


## Conclusão / Balanço

Outlook ligeiramente pessimista :(

**PROS**:
- Fácil de treinar os modelos mais pequenos;
    - Código reutilizável
- Permitem abstrair decisões arbitrárias no que toca a definição de heurísticas;
- Introduzir(?) conceito de sugestões/recomendações no RM

**CONS**:
- A qualidade dos resultados finais pode não render com base nos "custos" e esforços de implementação:
    - Full-text search do RM já é bastante sólido
    - Fácil de criar scripts para implementar "solução tradicional"
- Requerem dados de qualidade e já existentes;
- Provavelmente não podem ser partilhados entre clientes devido a questões de privacidades.



### Extra (Hipotético e pouco pensado):
- Tentar perceber se com isto podemos implementar uma "framework" para construção de "custom decision-trees" que possamos utilizar?
    - Composto por 'nodes'
    - Cada node tem um "decision algorithm" escolhido de entre X possíveis
    - Input (instance rm) -> output label/score/etc