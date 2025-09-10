## FuST-KGC: Fusing Sub-graph Structures and Textual Semantics for Knowledge Graph Completion
## [Project Page](placeholder) | Run Analysis Baseline

## 📖 Table of Contents <a href="#top">[Back to Top]</a>

- [Requirements](#requirements-)
- [How to Run](#how-to-run-)
- [Data layout](#Data-layout-)
- [WN18RR dataset](#WN18RR-dataset-)
- [FB15k-237 dataset](#fb15k237-dataset-)
- [Inference-only: get prediction files](#inference-)


## 🌠 Requirements <a href="#top">[Back to Top]</a> <a name="requirements-"></a>
All experiments are run with 2 4090(24GB) GPUs.
```bash
python>=3.7
torch>=1.6 (for mixed precision training)
transformers>=4.15
```

## 📄 How to Run <a href="#top">[Back to Top]</a> <a name="how-to-run-"></a>

It involves 3 steps: dataset preprocessing, model training, and model evaluation.

We also provide the predictions from our models in [result](result/) directory.

For WN18RR and FB15k237 datasets, we use files from [KG-BERT](https://github.com/yao8839836/kg-bert).


## 🛸 Data layout <a href="#top">[Back to Top]</a> <a name="Data-layout-"></a>
Place raw datasets under `data/` following these defaults:
```bash
** WN18RR**:
   `data/WN18RR/train.txt`, `data/WN18RR/valid.txt`, `data/WN18RR/test.txt`
   `data/WN18RR/wordnet-mlj12-definitions.txt`
** FB15k-237**:
   `data/FB15k-237/train.txt`, `data/FB15k-237/valid.txt`, `data/FB15k-237/test.txt`
   `data/FB15k-237/FB15k_mid2name.txt`, `data/FB15k-237/FB15k_mid2description.txt`
```
Outputs produced by preprocessing (JSON files) will be saved alongside the raw files.

## 🚀 WN18RR dataset <a href="#top">[Back to Top]</a> <a name="WN18RR-dataset-"></a>

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

## 🔥 FB15k-237 dataset <a href="#top">[Back to Top]</a> <a name="fb15k237-dataset-"></a>

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

## 🧪 Inference-only: get prediction files <a href="#top">[Back to Top]</a> <a name="inference-"></a>
The evaluation script will produce detailed per-sample predictions and metrics near the checkpoint directory, e.g.:
- `eval_test_forward_model_best.mdl.json`
- `eval_test_backward_model_best.mdl.json`
- `metrics_test_model_best.mdl.json`

