<<<<<<< HEAD
## FuST-KGC: Fusing Sub-graph Structures and Textual Semantics for Knowledge Graph Completion


## Requirements
* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15

All experiments are run with 2 4090(24GB) GPUs.

## How to Run

It involves 3 steps: dataset preprocessing, model training, and model evaluation.

We also provide the predictions from our models in [result](result/) directory.

For WN18RR and FB15k237 datasets, we use files from [KG-BERT](https://github.com/yao8839836/kg-bert).


### Data layout
Place raw datasets under `data/` following these defaults:
- **WN18RR**:
  - `data/WN18RR/train.txt`, `data/WN18RR/valid.txt`, `data/WN18RR/test.txt`
  - `data/WN18RR/wordnet-mlj12-definitions.txt`
- **FB15k-237**:
  - `data/FB15k-237/train.txt`, `data/FB15k-237/valid.txt`, `data/FB15k-237/test.txt`
  - `data/FB15k-237/FB15k_mid2name.txt`, `data/FB15k-237/FB15k_mid2description.txt`

Outputs produced by preprocessing (JSON files) will be saved alongside the raw files.

### WN18RR dataset

Step 1, preprocess the dataset
```
python preprocess.py \
  --task wn18rr \
  --workers 4 \
  --train-path data/WN18RR/train.txt \
  --valid-path data/WN18RR/valid.txt \
  --test-path  data/WN18RR/test.txt
```

Step 2, training the model and (optionally) specify the output directory 
```
python main.py \
  --task wn18rr \
  --pretrained-model distilbert \
  --train-path data/WN18RR/train.txt.json \
  --valid-path data/WN18RR/valid.txt.json \
  --model-dir checkpoints/wn18rr \
  --batch-size 128 \
  --epochs 50 \
  --use-amp
```

Step 3, evaluate a trained model
```
# Evaluate on test split (forward/backward and averaged metrics will be saved)
python evaluate.py \
  --task wn18rr \
  --valid-path data/WN18RR/test.txt.json \
  --eval-model-path checkpoints/wn18rr/model_best.mdl
```

Feel free to change the output directory to any path you think appropriate.

### FB15k-237 dataset

Step 1, preprocess the dataset
```
python preprocess.py \
  --task fb15k237 \
  --workers 4 \
  --train-path data/FB15k-237/train.txt \
  --valid-path data/FB15k-237/valid.txt \
  --test-path  data/FB15k-237/test.txt
```

Step 2, training the model and (optionally) specify the output directory 
```
python main.py \
  --task fb15k237 \
  --pretrained-model distilbert \
  --train-path data/FB15k-237/train.txt.json \
  --valid-path data/FB15k-237/valid.txt.json \
  --model-dir checkpoints/fb15k237 \
  --batch-size 128 \
  --epochs 10 \
  --use-amp
```

Step 3, evaluate a trained model
```
python evaluate.py \
  --task fb15k237 \
  --valid-path data/FB15k-237/test.txt.json \
  --eval-model-path checkpoints/fb15k237/model_best.mdl
```

### Inference-only: get prediction files
The evaluation script will produce detailed per-sample predictions and metrics near the checkpoint directory, e.g.:
- `eval_test_forward_model_best.mdl.json`
- `eval_test_backward_model_best.mdl.json`
- `metrics_test_model_best.mdl.json`

=======
# FuST-KGC
我们稍后会公开所有代码和数据集
>>>>>>> 26a284763d69da296d3bb69a7f5aae643f630b90

