# CodeHID

CodeHID is a generative code retrieval framework that reformulates code retrieval from flat candidate matching to coarse-to-fine semantic address generation.

It has two core components:

- Pseudo-Neighbor Guided DocID Learning: builds a globally static hierarchical index with multi-level residual quantization and k-nearest-neighbor pseudo-supervision.
- Dual-Phase DocID Generation Guidance: improves training with hard negatives and rank distillation, and improves inference with candidate constraints and prefix-aware decoding.

## Benchmarks

- CoSQA
- ProCQA

## Qualitative Examples

See [Hierarchical DocID Generation Examples](examples/docid_generation_cases.md) for three real cases from the CoSQA development split illustrating constrained DocID generation, shared-prefix branching, and fine-grained competition within the learned hierarchy.

## Repository Layout

```text
CodeHID/
├── codehid/
├── examples/                    # Qualitative DocID generation examples
├── stage1_docid_build/
├── train_configs/
├── train.py
├── eval.py
├── run_3gpu.sh
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Default Settings

The KNN-related defaults are built into `stage1_docid_build/rqvae/train_rqvae.py`:

- `WORK_DIR` can be set to any local path you choose.
- `KNN_K=8`
- `KNN_BATCH_SIZE=2048`
- `KNN_LOSS_WEIGHT=0.03`
- `KNN_LOSS_SCHEDULE=linear_decay_to_zero`
- `KNN_DECAY_START_RATIO=0.3`
- `KNN_DECAY_END_RATIO=0.7`
- `KNN_TEMPERATURE=0.35`
- `KNN_PREFIX_WEIGHTS=0.8 0.35 0.12 0.03`
- `KNN_AUTO_THRESHOLDS=1`
- `KNN_THRESHOLD_QUANTILES=0.85 0.95 0.99 0.999`
- `KNN_NUM_NEGATIVES=4`
- `KNN_NEGATIVE_WEIGHT=0.6`

## Main Workflows

- Build hierarchical DocIDs:
  - `WORK_DIR=<your_work_dir> USE_KNN_PREFIX_LOSS=1 KNN_K=8 KNN_BATCH_SIZE=2048 KNN_LOSS_WEIGHT=0.03 KNN_LOSS_SCHEDULE=linear_decay_to_zero KNN_DECAY_START_RATIO=0.3 KNN_DECAY_END_RATIO=0.7 KNN_TEMPERATURE=0.35 KNN_PREFIX_WEIGHTS="0.8 0.35 0.12 0.03" KNN_AUTO_THRESHOLDS=1 KNN_THRESHOLD_QUANTILES="0.85 0.95 0.99 0.999" KNN_NUM_NEGATIVES=4 KNN_NEGATIVE_WEIGHT=0.6 bash stage1_docid_build/run_stage1_pipeline.sh cuda:0 "$WORK_DIR" <encoder_path> <train_json> <dev_json>`
- Build GraphCodeBERT embeddings:
  - `python stage1_docid_build/encode_code_docs.py --docs_jsonl <work_dir>/docs.jsonl --output_npy <work_dir>/docs.emb-graphcodebert.npy --model_name <graphcodebert_path> --text_mode code --batch_size 32 --max_length 512 --device cuda:0`
- Train the retriever:
  - `bash run_3gpu.sh`
- Evaluate a checkpoint:
  - `python eval.py --ckpt-path <ckpt> --dataset-path <valid.jsonl> --docid-vocab-path <doc_index_to_docid.json> --dual-phase-guidance-infer-doc-emb-path <docs.emb-graphcodebert.npy> --constrained-decode --dual-phase-guidance-infer-enable --num-beams 20`

## Notes

- DocIDs are globally static and query-independent.
- The hierarchical prefix structure is designed to preserve semantic locality.
- This repository is intended for anonymous release.
