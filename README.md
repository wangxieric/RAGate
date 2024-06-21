# RAGate: Adaptive Retrieval-Augmented Generation for Conversational Systems
This repository includes the code and resources for our paper entitled "Adaptive Retrieval-Augmented Generation for Conversational Systems" for reproducible findings.
<div align="center">
  
  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

## Links
- [Full Data Generation](#full-data-generation)
  
## Full Data Generation
RAGate leverages the KETOD dataset, which extends the Google SGD dataset with additional human labels on adaptive knowledge augmentation with external knowledge.
To generate the full KETOD dataset, run gen_ketod_data.py from the [src](src) folder.
```
  python gen_ketod_data.py 
```

