# AI_genderbiases: Debiasing Urdu Word Embeddings

## Overview
This project focuses on mitigating the negative impacts of gender biases in pre-trained NLP models, specifically for the Urdu language. Through the provided scripts and methodologies, users can debias Urdu word embeddings created using a skip-gram model. Our approach involves adjusting word embeddings to reduce gender bias.

## Urdu Word Embeddings
The Urdu Word Embedding created for this project utilizes the skip-gram model technique. For an in-depth look into the embedding creation process, please refer the paper: https://aclanthology.org/L18-1155.pdf

## Scripts
The repository includes Python scripts designed to learn and mitigate gender biases in word embeddings:

### `learn_gender_specific.py`
Expands a seed set of gender-specific words into a comprehensive list based on given word embeddings.

Run Command: _python learn_gender_specific.py ../embeddings/urduvec_140M_100K_300d.bin 50000 ../data/urdu_gender_specific_seed.json urdu_gender_specific_full.json_

### `debias.py`
Adjusts a word embedding by considering sets of gender pairs, gender-specific words, and pairs to equalize, producing a new, debiased word embedding.

Run Command: _python debias.py ../embeddings/urduvec_140M_100K_300d.bin ../data/urdu_definitional_pairs.json ../data/urdu_gender_specific_full.json ../data/urdu_equalize_pairs.json ../debiased_embeddings_140M_100K_300d.bin_

## Data Files
The project utilizes various data files to identify and mitigate gender bias effectively:

### `gender_specific_seed.json` 
Contains 101 gender-specific words.
### `gender_specific_full.json`
Expanded list of 650 gender-specific words.
### `definitional_pairs.json`
20 pairs of words defining the gender direction.
### `equalize_pairs.json`
Crowdsourced male-female pairs of words representing gender direction.

## How to Use
For detailed instructions on setting up the environment, running the scripts, and employing the debiased embeddings for further NLP tasks, please refer to the demo notebook.

## Contributing
Contributions are welcome! If you have suggestions for improving the debiasing process or other aspects of the project, please review our contribution guidelines and submit a pull request.

## Contact
For questions or collaboration inquiries, please open an issue on GitHub.


