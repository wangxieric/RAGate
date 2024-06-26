# RAGate: Adaptive Retrieval-Augmented Generation for Conversational Systems
This repository includes the code and resources for our paper entitled "Adaptive Retrieval-Augmented Generation for Conversational Systems" for reproducible findings.
<div align="center">
  
  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

## Links
- [Full Data Generation](#full-data-generation)
- [Knowledge Retrieval Model Development](#knowledge-retrieval-model-development)
  
## Full Data Generation
RAGate leverages the KETOD dataset, which extends the Google SGD dataset with additional human labels on adaptive knowledge augmentation with external knowledge.
To generate the full KETOD dataset, run gen_ketod_data.py from the [src](src) folder.
```
  python gen_ketod_data.py 
```

## Knowledge Retrieval Model Development
RAGate implements two retrieval models (TF-IDF and BERT/RoBERTa-ranker). 
It is required to first process the generated data into proper structures for knowledge retrieval. 
If BERT/RoBERTa-ranker is used, you need to train a ranking model after that. The example procedure is as follows:

```
- go to src/kg_select
- python process_data.py --data train.py / dev.py / test.py
- python train.py (update config.py if needed)
```
We share the trained BERT-ranker and can be downloaded via the following link: [trained-ranker](https://drive.google.com/drive/folders/1LSg71IicaLCwjOVFPcJeBanMNl7zTvS-?usp=drive_link)
