# Finetune Gemma LLM on Medical Question Answering Dataset (MedQuAD)

![process](kaggleX%20Chatbot.png)

Steps:
1. Source for dataset. [MedQuAD (2019)](https://github.com/abachaa/MedQuAD) dataset was used
2. Extract and clean data [data_extraction](/data_extraction.ipynb)
3. Format data for prompting [formating](/prompt%20engineering.ipynb)
4. Finetune Gemma LLM using Keras JAX on [dataset](/gamma-finetune-with-medquad-data-keras-jax%20(1).ipynb) and convert model to Hugginface Transformer
5. Deploy finetuned model on [Vertex AI](/Deploy.ipynb)


Findings:
1. The Vertex AI deployed models generated output wasnt efficient as compared to the non-deployed model's output - see [deploy](/Deploy.ipynb)
2. The inference generation was a bit slow, tried using `cuda` for GPU and `mps` for Apple backend processing. This resulted in a bit faster inference generation, next is to attempt quantization -[deploy](/deploy/generate.py)


Next steps:
1. Finetune Gemma using Keras and Tokenizers and quantization. 
2. Research: Is Finetuned model effective for student evaluation using Rubric score.


References:
        @ARTICLE{BenAbacha-BMC-2019,    
            author    = {Asma {Ben Abacha} and Dina Demner{-}Fushman},
            title     = {A Question-Entailment Approach to Question Answering},
            journal = {{BMC} Bioinform.}, 
            volume    = {20},
            number    = {1},
                pages     = {511:1--511:23},
            year      = {2019},
        url       = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4}
            }   
