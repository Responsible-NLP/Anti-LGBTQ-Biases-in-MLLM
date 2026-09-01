# Evaluating and Mitigating Anti-LGBTQ Biases in German and Multilingual Language Models

## Abstract
While gender and racial biases in language models have been widely studied, anti-LGBTQ biases remain underexplored, particularly beyond English. Existing benchmarks often do not capture cultural and linguistic variation and rely on gender representations. This paper introduces a multilingual German-English benchmark dataset for the evaluation of anti-LGBTQ biases in language models. It combines community-sourced stereotypes from German-speaking queer individuals with a German translation of WinoQueer. The data is used to evaluate eight language models across sizes and architectures and explore mitigation through fine-tuning on community and progressive media content. Results show that language models reproduce anti-queer stereotypes, with variation across identities and models. Differences between the translated and community-based data highlight the importance of cultural adaptation for multilingual bias evaluation. Fine-tuning reduces bias on average, but not consistently across models and identities. Warning: This text contains examples of anti-queer hateful language and stereotypes.

## Content of the Repository

* [corpora](corpora/): Contains three corpora:
    * [winoqueerDE](corpora/winoqueerDE.csv): German Translation of the WinoQueer Corpus
    * [community](corpora/community.csv): Novel corpus derived from a survey conducted with more than 100 German speaking queer individuals
    * [mastodon_queer_german_posts](corpora/mastodon_queer_german_posts.csv): Corpus of queer-friendly Mastodon posts used for bias mitigation through fine-tuning
* [code](code/): The code used for the creation of the corpora, the bias eveluation of the models, and their fine-tuning

## Citation
````
@misc{morch2026evaluatingmitigatingantilgbtqbiases,
      title={Evaluating and Mitigating Anti-LGBTQ Biases in German and Multilingual Language Models}, 
      author={Melina Morch and Daniel Braun},
      year={2026},
      eprint={2608.30884},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.30884}, 
}
````
