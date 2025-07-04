# FLD Generator
![framework_overview](./images/framework_overview.PNG)

This repository includes the code for generating the FLD corpus.  

See [the entry-point repository](https://github.com/hitachi-nlp/FLD.git) about the whole FLD project.




## Release Branches  (READ CAREFULLY to determine which branch suits you)
* **(New!)** `NeurIPS_2024` branch (2024-12)
    - We released the code for generating FLDx2 (Formal Logic Deduction Diverse).
* `NLP_2024_KOBE_BEEF` branch (2024-01-24) 
    - Release at LREC-COLING 2024 and 言語処理学会 2024.
    - Now capable of generating Japanese corpora (JFLD).
    - Slight changes in the corpus schema.
    - This branch and the generated corpora might not be compatible with older branches of related repositories.
* `ICML_2023` branch (2023-08-22)
    - Initial release at ICML 2023.
    - This is version 2.0 of FLD corpora. See the Appendix H of [our paper](https://arxiv.org/abs/2308.07336) for details.




## Installation
The code has been tested with Python 3.11.5.
```console
pip install -r ./requirements/requrements.txt
pip install wheel
export PYTHONPATH=`pwd -P`:$PYTHONPATH
```

### Additional Resources Required
* For Japanese FLD
    ```console
    ./download_scripts/00.download_JFLD_resources.sh
    ```
* For knowledge FLD
    ```console
    ./download_scripts/00.download_knowledge_banks.sh
    ```




## How to Generate FLDx2 (Formal Logic Deduction Diverse) Corpus
We create 100k examples of FLDx2 using the generation script `./scripts/create_corpus.py`.
We recommend to parallelize the run, as the computation is large, roughly estimated as a few thousand CPU hours.

1. Create 95k examples with less diverse natural language expressions:
```console
python ./scripts/create_corpus.py \
    {output_dir} \
    {num_examples} \
    --translation-config old-thing.v1 \
    --generate-stem-steps-range '[1, 3]' \
    --extend-branches-steps-range '[0, 5]' \
    --argument-config ./configs/arguments/predicate/specified/axioms/ \
    --argument-config ./configs/arguments/propositional/axioms/ \
    --argument-config ./configs/arguments/predicate/specified/references/ \
    --argument-config ./configs/arguments/propositional/references/ \
    --argument-config ./configs/arguments/predicate/quantified/references/ \
    --argument-config ./configs/arguments/predicate/specified/theorems \
    --argument-config ./configs/arguments/propositional/theorems \
    --argument-config ./configs/arguments/predicate/quantified/theorems \
    --negation-arguments-weight 0.1 \
    --theorem-tree-prob 0.15 \
    --theorem-arguments-factor 0.01 \
    --theorem-arguments-weight-adjustment-subset G_MP.syllogism.contraposition.interchangeability \
    --reference-tree-prob 0.2 \
    --num-workers 10 \
    --seed 0
```

2. Create 5k examples with more diverse natural language expressions (which requires more computation):
```console
python ./scripts/create_corpus.py \
    {output_dir} \
    {num_examples} \
    --translation-config thing_person.v2 \
    --generate-stem-steps-range '[1, 3]' \
    --extend-branches-steps-range '[0, 5]' \
    --argument-config ./configs/arguments/predicate/specified/axioms/ \
    --argument-config ./configs/arguments/propositional/axioms/ \
    --argument-config ./configs/arguments/predicate/specified/references/ \
    --argument-config ./configs/arguments/propositional/references/ \
    --argument-config ./configs/arguments/predicate/quantified/references/ \
    --argument-config ./configs/arguments/predicate/specified/theorems \
    --argument-config ./configs/arguments/propositional/theorems \
    --argument-config ./configs/arguments/predicate/quantified/theorems \
    --negation-arguments-weight 0.2 \
    --theorem-tree-prob 0.15 \
    --theorem-arguments-factor 0.01 \
    --theorem-arguments-weight-adjustment-subset G_MP.syllogism.contraposition.interchangeability \
    --reference-tree-prob 0.1 \
    --num-workers 10 \
    --seed 261
```

3. Concatenate the above examples to create "raw" FLDx2 corpus.

4. Make "prompt-output" pairs from the corpus, following [FLD-task](https://github.com/hitachi-nlp/FLD-task).
