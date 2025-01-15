# NanoGPT4Rec: Transformer-based Recommendation System

![](assets/nanogpt4rec.jpg)

This repository extends [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) implementation to create a transformer-based recommendation system.

ProductGPT adapts the GPT architecture for sequential recommendation tasks with support for cold-start scenarios.

## Features

- Sequential recommendation using transformer architecture
- Handles cold-start problems through context features
- Time-aware recommendations with temporal patterns
- Support for both CPU and GPU (CUDA/MPS) training
- Evaluation using standard recommendation metrics (Hit Rate@10, NDCG@10)

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- Basic dependencies: numpy, pandas, scikit-learn

```bash
pip install -r requirements.txt
```

### Mac M1/M2/M3 Setup
The code automatically uses MPS (Metal Performance Shaders) on Apple Silicon:
1. Install dependencies from requirements.txt
2. Run training with --device=mps flag

## Dataset

Currently supports MovieLens-1M dataset.

```
data/
  movielens-1m/
    ratings.dat
```

## Training

### Basic Training
```bash
# For CPU
python train.py --device=cpu --batch_size=32 --block_size=50

# For GPU
python train.py --device=cuda --batch_size=32 --block_size=50

# For Mac M1/M2/M3/M4
python train.py --device=mps --batch_size=32 --block_size=50
```

### Configuration Parameters

- `--batch_size`: Number of sequences per batch (default: 32)
- `--block_size`: Maximum sequence length (default: 50)
- `--n_layer`: Number of transformer layers (default: 6)
- `--n_head`: Number of attention heads (default: 8)
- `--n_embd`: Embedding dimension (default: 384)
- `--dropout`: Dropout rate (default: 0.1)
- `--learning_rate`: Learning rate (default: 1e-4)

## Model Architecture

ProductGPT extends the base GPT architecture with:
1. User embeddings
2. Temporal context features:
   - Time of day (cyclic encoding)
   - Day of week (cyclic encoding)
   - Time deltas between interactions
3. Rating information
4. Transformer backbone for sequential modeling

## Evaluation Metrics

The model is evaluated using standard recommendation metrics:
- Hit Rate@10: Proportion of times the correct item appears in top-10 recommendations
- NDCG@10: Normalized Discounted Cumulative Gain for top-10 recommendations
- Validation Loss

## Example Usage

```python
from model import ProductGPT, ProductGPTConfig

# Initialize model
config = ProductGPTConfig(
    block_size=50,
    n_layer=6,
    n_head=8,
    n_embd=384,
    max_products=10000,
    max_users=1000,
    n_context_features=6,
)

model = ProductGPT(config)

# Get recommendations
recommendations, scores = model.get_recommendations(
    context_features=current_context,
    user_id=user_id,
    product_history=past_interactions
)
```

## Acknowledgments

This implementation is built upon [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. The original GPT implementation provides the foundation for this recommendation system.

## License

MIT License - see LICENSE file for details.
