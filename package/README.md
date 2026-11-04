# MedSAE

**Sparse autoencoders for interpretable and transferable medical retrieval.** Reference implementation of the ML4H
2026 paper *Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders* (Gaëtan Brison and
Diane Bouchacourt).

![MedSAE overview](../.github/assets/medsae_overview.png)

MedSAE trains a TopK sparse autoencoder (SAE) on the frozen embeddings of a text encoder, with a reconstruction loss
and an InfoNCE loss that keeps queries close to their relevant documents. Documents are then ranked by cosine
similarity of codes with at most k active features, and every feature can be described from the texts that activate
it: top snippets, TF IDF terms and a topic written by an Interpreter LLM. These descriptions explain single
retrievals, and they transfer to corpora the SAE never saw.

> **Warning:** MedSAE is in active development and its API may change before version 1.0. It is a research tool:
> do not use it for diagnosis, treatment or any other clinical decision.

## Install

```bash
pip install medsae
pip install "medsae[llm]"        # optional: LLM descriptions through any OpenAI compatible server
pip install "medsae[local-llm]"  # optional: local LLM such as google/medgemma-4b-it
```

MedSAE needs Python 3.11 or newer. A GPU is advised for CURE; the small examples run on a laptop CPU.

## Quickstart

```python
from medsae import Embedder, Interpreter, MedSAE, Trainer, TrainingConfig, evaluate, load_dataset

dataset = load_dataset("nfcorpus")                                # cure, miriad, r2med_biology, ...
data = Embedder("BAAI/bge-small-en-v1.5").embed_dataset(dataset)  # frozen encoder

sae = MedSAE(d_in=data.dim, n_latents=3 * data.dim, k=32, embedder=data.embedder)
sae = Trainer(sae, TrainingConfig(), output_dir="runs/nfcorpus/sae").fit(data).best_model  # paper settings

print(evaluate(None, data, split="test"))  # dense embeddings
print(evaluate(sae, data, split="test"))   # sparse codes: ndcg@10, recall@100, mrr@10

interpreter = Interpreter(sae, data, split="train")
for card in interpreter.describe(interpreter.select_features(n=3)):
    print(card.to_markdown())
```

The same steps from the command line:

```bash
medsae embed --dataset nfcorpus --model BAAI/bge-small-en-v1.5 --output runs/nfcorpus/embeddings
medsae train --dataset nfcorpus --embeddings runs/nfcorpus/embeddings --k 32 --output-dir runs/nfcorpus/sae
medsae evaluate --dataset nfcorpus --embeddings runs/nfcorpus/embeddings --sae runs/nfcorpus/sae
medsae interpret --dataset nfcorpus --embeddings runs/nfcorpus/embeddings --sae runs/nfcorpus/sae \
    --output runs/nfcorpus/features --n-features 20
```

## What is inside

* `medsae.load_dataset`: CURE, NFCorpus, MIRIAD and the three R2MED subsets, downloaded from the Hugging Face Hub
  at pinned revisions with the splits of the paper; `medsae.load_beir_folder` for your own data.
* `medsae.Embedder`: sentence transformers encoders; queries of the BGE v1.5 models get the retrieval instruction
  of their model card, the one the paper used for BGE large, and the embeddings record every setting.
* `medsae.MedSAE`: the TopK SAE, saved and shared through the Hugging Face Hub (`save_pretrained`, `push_to_hub`,
  `from_pretrained`, `medsae.load_pretrained`).
* `medsae.Trainer`: the paper objective, sampler and checkpoint selection. The defaults are the values the paper
  holds fixed, and one point of its grid for the InfoNCE weight and temperature, which it selects per k.
* `medsae.evaluate` and `medsae.search`: retrieval on sparse codes, NDCG@10 computed as in the paper.
* `medsae.Interpreter`: feature selection, feature cards, Interpreter and Activation Prediction LLMs (OpenAI
  compatible servers, local Transformers models), and retrieval explanations as in Figure 2 of the paper.
* `medsae` command line: `datasets`, `pretrained`, `embed`, `train`, `evaluate`, `interpret`, `explain`, `push`,
  `convert`.

## Pretrained sparse autoencoders

Checkpoints trained on CURE are planned under the [`doctolib-lab`](https://huggingface.co/doctolib-lab)
organisation, one repository per encoder and sparsity k, such as `doctolib-lab/medsae-bge-large-en-v1.5-k64`. None
is published yet; `medsae pretrained` lists what is available. Weights trained on CURE are `cc-by-nc-4.0`
(noncommercial), the license of CURE.

## Results from the paper

NDCG@10 (%) with `BAAI/bge-large-en-v1.5`, SAEs trained on CURE, mean over three seeds (Table 1 of the paper):

| BGE | CURE | NFCorpus | MIRIAD | R2MED Bioinformatics | R2MED Biology | R2MED Medical Sciences |
| --- | --- | --- | --- | --- | --- | --- |
| original | 53.06 | 38.17 | 77.24 | 27.04 | 12.72 | 27.78 |
| k = 64 | 57.07 | 31.89 | 68.75 | 22.50 | 8.30 | 22.72 |
| k = 128 | 59.35 | 34.48 | 71.83 | 27.42 | 9.76 | 26.60 |
| k = 256 | 60.50 | 35.61 | 73.85 | 28.10 | 10.06 | 29.05 |
| dense (k = 1024) | 61.87 | 36.62 | 75.05 | 27.21 | 8.37 | 27.33 |

## Links

* Source code, paper code and full documentation: <https://github.com/doctolib-lab/medsae>
* Examples: <https://github.com/doctolib-lab/medsae/tree/main/package/examples>
* Issues: <https://github.com/doctolib-lab/medsae/issues>
* Contributing: <https://github.com/doctolib-lab/medsae/blob/main/CONTRIBUTING.md>

## Citation

```bibtex
@inproceedings{brison2026explanations,
  title     = {Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders},
  author    = {Brison, Ga{\"e}tan and Bouchacourt, Diane},
  booktitle = {Machine Learning for Health (ML4H)},
  year      = {2026}
}
```

## License

The code is released under the Apache 2.0 license. Datasets keep their own licenses and are downloaded from the
Hugging Face Hub at pinned revisions (CURE is `cc-by-nc-4.0`, noncommercial). SAE weights trained on CURE are
`cc-by-nc-4.0`.
