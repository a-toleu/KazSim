


# KazSim - Kazakh Text Simplification

A text simplification library using fine-tuned LLaMA models with Unsloth for efficient training.

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start

### Training
```python
python scripts/train.py --config configs/training_config.yaml
```

### Testing
```python
python scripts/test.py --model_path models/your_model --input "Your complex text here"
```


# KazSim Dataset

This repository contains a parallel dataset for the task of text simplification in the Kazakh language. The dataset consists of pairs of complex and simplified Kazakh sentences, curated to support research and development in natural language processing (NLP), especially for low-resource and morphologically rich languages.

Key Features:

Parallel Data: Each entry includes a complex (original) Kazakh sentence and its simplified counterpart.

Size: The dataset contains 8700 sentence pairs for training set and 500 sentence pairs for test set. 

Sources: Complex sentences are sourced from various authentic Kazakh texts, including literature. Simplified versions are either manually created by semi-automatically generated.

Intended Use: Suitable for training, evaluation, and benchmarking of text simplification models for Kazakh.


## Features
- 4-bit quantization for efficient training
- WandB integration for experiment tracking
- Modular architecture for easy customization
- Support for multiple instruction templates

## License
cc-by-nc-4.0
