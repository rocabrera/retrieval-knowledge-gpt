# retrieval-knowledge-gpt
Framework to understand LLMs behavior


# Execution

Steps:
- Configure config file [here](./src/conf/config.yaml)
- In the root of the repository execute: ```python3 src/main.py```

# Predictor models

- GPT:
    - Id: ```gpt-3.5-turbo```
    - Name: ``` gpt3.5_turbo```

- Flan T5:
    - Id: ```google/flan-t5-xl```
    - Name: ```flan_t5_xl```


# Sugestões
 
Ver na docs do OpenAI:
- Max de tokens PROMPT -> 4096.
- Max de tokens (generate).
- Olhar temperatura.


# TODO List 

- Improve console logging
- Get LLM should receive config params which must be traceable 