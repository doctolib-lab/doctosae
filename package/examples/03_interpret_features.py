"""Interpret the features of a MedSAE: feature cards, Activation Prediction scores and one explanation.

By default the language model is ``FakeLLM``, a deterministic stand in that needs no download: the pipeline
runs end to end in seconds, but its labels and scores are placeholders. For real interpretations pass
``--llm openai`` (any OpenAI compatible server such as vLLM or Ollama) or ``--llm transformers`` (a local
model; the paper used ``google/medgemma-4b-it``, which is gated: accept its terms on the Hub, then run
``hf auth login``).

    python examples/03_interpret_features.py                     # the SAE trained by 01_quickstart.py
    python examples/03_interpret_features.py --llm transformers  # MedGemma, as in the paper (GPU advised)
    python examples/03_interpret_features.py --llm openai --llm-model qwen2.5:7b-instruct \\
        --base-url http://localhost:11434/v1                     # a model served by Ollama
    python examples/03_interpret_features.py --ood r2med_biology # also score the cards out of domain

The protocol of Table 3 of the paper interprets a CURE checkpoint of ``02_reproduce_cure.py`` on the CURE
training passages and queries. Every qualifying feature is described (``--n-features 0``), in the order of
``select_features(theme_column="source_folder")``, most theme specific first. That order drives both seeded
draws of the scoring (the evaluation rows, then the pairs), so another order judges other texts. The cards are
then scored in domain and once per out of domain dataset:

    python examples/03_interpret_features.py --sae runs/cure/bge_k64_m3072_seed42 --dataset cure \\
        --embeddings runs/cure/embeddings/cure --theme-column source_folder --n-features 0 \\
        --llm transformers --ood nfcorpus miriad r2med_bioinformatics r2med_biology r2med_medical_sciences
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from medsae import (
    EmbeddedDataset,
    Embedder,
    Interpreter,
    MedSAE,
    OpenAICompatibleLLM,
    TransformersLLM,
    load_dataset,
    load_pretrained,
    search,
)
from medsae.interpret import LLM, FakeLLM, ScoreReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe, score and use the features of a MedSAE.")
    parser.add_argument("--sae", default="runs/quickstart/sae", help="MedSAE folder or Hub repository")
    parser.add_argument("--dataset", default="nfcorpus", help="dataset to interpret on (default: nfcorpus)")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("runs/quickstart/embeddings"),
        help="its saved embeddings, if any",
    )
    parser.add_argument(
        "--ood",
        nargs="*",
        default=[],
        help="out of domain datasets to score the cards on, such as r2med_biology",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=20,
        help="features to describe, 0 for all as the paper (default: 20)",
    )
    parser.add_argument(
        "--theme-column", help="corpus column of themes that orders the features (source_folder for CURE)"
    )
    parser.add_argument(
        "--llm", choices=("fake", "openai", "transformers"), default="fake", help="LLM backend"
    )
    parser.add_argument("--llm-model", help="served model name (openai) or Hub id (transformers)")
    parser.add_argument("--base-url", help="URL of the OpenAI compatible server")
    parser.add_argument("--output", type=Path, default=Path("runs/quickstart/features"), help="output folder")
    parser.add_argument("--device", help="cuda, mps or cpu (default: the best available)")
    return parser.parse_args()


def build_llm(args: argparse.Namespace) -> LLM:
    if args.llm == "openai":
        if not (args.llm_model and args.base_url):
            raise SystemExit(
                "--llm openai needs --llm-model and --base-url, the server the texts are sent to"
            )
        return OpenAICompatibleLLM(args.llm_model, base_url=args.base_url)
    if args.llm == "transformers":
        return TransformersLLM(args.llm_model) if args.llm_model else TransformersLLM()
    logging.warning("FakeLLM: the labels and scores below are placeholders that only exercise the pipeline.")
    return FakeLLM()


def table(report: ScoreReport) -> str:
    """Mean Pearson r, mean AUROC and counts per interpretation type."""
    return report.to_frame().to_string(float_format="{:.3f}".format)


def embedded(name: str, sae: MedSAE, folder: Path | None, device: str | None) -> EmbeddedDataset:
    """Embeddings of ``name`` by the encoder the SAE was trained on (reused when ``folder`` holds them)."""
    dataset = load_dataset(name)
    embedder = Embedder.from_sae(sae, device=device)
    if folder is not None and (folder / "meta.json").exists():
        try:
            saved = EmbeddedDataset.load(folder, dataset)
        except ValueError as error:  # other texts under the same ids
            logging.warning("Embedding %s again: %s", name, error)
        else:
            if saved.embedding == embedder.config:
                return saved
            logging.warning("%s holds embeddings of other encoder settings: embedding again", folder)
    return embedder.embed_dataset(dataset)


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("medsae").setLevel(logging.INFO)  # progress of medsae, warnings only from libraries
    sae = MedSAE.from_pretrained(args.sae) if Path(args.sae).exists() else load_pretrained(repo_id=args.sae)
    data = embedded(args.dataset, sae, args.embeddings, args.device)
    llm = build_llm(args)

    # 1. Features that fire on at least 5 training documents and 5 training queries, and their cards: top
    #    documents and queries, negatives, TF IDF terms, and the topic, description and keywords of the LLM.
    interpreter = Interpreter(sae, data, split="train", device=args.device)
    features = interpreter.select_features(n=args.n_features or None, theme_column=args.theme_column)
    cards = interpreter.describe(features, llm=llm)
    interpreter.save_cards(cards, args.output / "cards.json")
    for card in cards[:3]:
        print("\n" + card.to_markdown(max_chars=200))

    # 2. Activation Prediction: from a card alone, can the LLM tell where the feature fires? Mean Pearson r
    #    and AUROC per interpretation type, in domain, then on a corpus the SAE never saw.
    print("\nIn domain:\n" + table(interpreter.score(cards, llm)))
    for name in args.ood:
        ood = embedded(name, sae, None, args.device)
        print(f"\nOut of domain ({name}):\n" + table(interpreter.score(cards, llm, data=ood)))

    # 3. One retrieval explained: the features whose contributions to the cosine similarity differ most
    #    between the passages ranked first and tenth for the first test query (Figure 2 of the paper).
    split = "test" if "test" in data.queries else data.dataset.splits[-1]
    indices, _ = search(sae, data.queries[split][:1], data.corpus, top_k=10, device=args.device)
    query = data.dataset.queries[split]["text"].iloc[0]
    doc_a, doc_b = (data.dataset.corpus["text"].iloc[row] for row in indices[0, [0, 9]])
    print("\n" + str(interpreter.explain(query, doc_a, doc_b, cards=cards)))


if __name__ == "__main__":
    main()
