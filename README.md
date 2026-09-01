# Toxic Comment Classification (Jigsaw Dataset)

Multi-label classification of Wikipedia talk-page comments into six toxicity categories (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`), comparing a TF-IDF + Logistic Regression baseline against a from-scratch BiLSTM neural network with proper threshold tuning, error analysis, and an identity-term bias audit.

**Start here: [`toxic_comment_classification.ipynb`](toxic_comment_classification.ipynb)**: the notebook contains the full walkthrough (EDA, both models, comparison, error analysis, bias check, conclusions) with all plots and results already run and rendered inline, so it's readable without re-executing anything.

## Headline results

The simple baseline is competitive with (and on tuned F1 slightly ahead of) the neural network. That's a genuinely useful finding: added model complexity doesn't automatically win here. (The BiLSTM trains from scratch each run with no fixed seed guarantee across environments, so its exact numbers will drift slightly between runs.)

**Bias finding:** both models falsely flag genuinely non-toxic comments that merely mention being gay/lesbian/queer as toxic ~5x more often than their overall false positive rate. It is a known issue with this dataset (see notebook Section 8).

## Folder structure

```
.
├── toxic_comment_classification.ipynb   # Main deliverable: full analysis, run this first
├── requirements.txt                     # Core dependencies
├── predict.py                           # CLI: classify new text with the trained model
├── data_utils.py                        # Tokenizer, vocab, PyTorch Dataset (used by notebook + predict.py)
├── model.py                             # BiLSTM architecture (used by notebook + predict.py)
├── artifacts/                           # Everything needed to use the trained model
│   ├── bilstm_model.pt                  # trained BiLSTM weights
│   ├── bilstm_vocab.json                # vocabulary the model was trained with
│   ├── bilstm_config.json               # architecture hyperparameters
│   └── tuned_threshold_results.json     # per-label decision thresholds for predict.py
└── data/                                # Kaggle dataset (add these yourself, see below)
    ├── train.csv
    ├── test.csv
    └── test_labels.csv
```

All plots (EDA, model comparison, ROC curves, bias check) render inline in the notebook itself and aren't saved as separate image files. Open the notebook to see them.

## Setup

```PowerShell
pip install -r requirements.txt
```

This installs the CPU-only build of PyTorch (via the `--extra-index-url` in `requirements.txt`). Nnothing in this project needs a GPU. If you already have `torch` installed and hit a DLL/import error on Windows, it's usually because a CUDA-enabled build got installed without a matching NVIDIA setup; run `pip uninstall torch -y` first, then re-run the command above.

Download the dataset from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data and place `train.csv`, `test.csv`, `test_labels.csv` in a `data/` subfolder next to this README (create the folder if it doesn't exist).

## Running things

**Open the notebook**: it already has all outputs saved, but you can re-run it:

The BiLSTM section trains from scratch every time the notebook runs. The trained weights are saved to `artifacts/bilstm_model.pt` at the end of that section and are what `predict.py` uses afterward.

**Classify new text from the command line** (uses the checkpoint already saved in `artifacts/` from the last notebook run, no need to reopen the notebook unless you want to retrain):

```PowerShell
python predict.py "Some comment text here"
python predict.py --interactive
```

## Limitations (see notebook Section 9 for full discussion)

- BiLSTM trained on a class-balanced subsample for CPU-only compute reasons
- Identity-term bias is real and unmitigated in both models
- Rare classes (`threat`, `identity_hate`) remain hard for both models
- Trained on 2017-2018 Wikipedia comments; untested on other platforms/slang

A DistilBERT fine-tuning sketch (for further improvement on a GPU) is included as a reference code block at the end of the notebook.
