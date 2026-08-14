# RSICD-Captioner

Single-sentence captioning for satellite and aerial imagery, built from scratch in PyTorch.

Given an overhead image, the model produces one sentence describing what is in it — land use, structures, scene layout. Four encoder/decoder combinations are implemented and compared head to head, followed by seven ablations that isolate specific architectural choices.

```
input:  a 224×224 overhead image
output: "many buildings are around a large playground ."
```

---

## What's here

| | |
|---|---|
| **Encoders** | ResNet-18 (11.4M params) and MobileNet-v2 (2.9M params), both ImageNet-pretrained |
| **Decoders** | 1-layer LSTM, and a 4-layer Transformer decoder with image cross-attention |
| **Dataset** | [RSICD](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset) — 10,921 images, 5 human captions each |
| **Metrics** | BLEU-4, METEOR, caption length, degenerate-repetition rate |
| **Explainability** | Grad-CAM over the encoder's last conv block |
| **Experiments** | 7 ablations: backbones, rotation invariance, vision-text interface, memory tokens, dropout placement, length penalty |

Everything — tokenizer, dataset, encoder, both decoders, training loop, metrics — is written directly against PyTorch. No captioning framework is used.

---

## Results

Trained for 5 epochs on a single Tesla P100 (16 GB), fp16 mixed precision, effective batch size 512. Metrics are on a validation subset.

### Model comparison

| Model | Params | Best val loss | BLEU-4 | METEOR | Repetition | Inference |
|---|---|---|---|---|---|---|
| ResNet+LSTM | 15.9M | 3.246 | 0.003 | 0.217 | 60.1% | 24 ms |
| MobileNet+LSTM | 7.4M | 3.237 | 0.006 | 0.204 | 17.8% | 23 ms |
| ResNet+Transformer | 30.6M | 2.027 | 0.078 | 0.400 | 0.4% | 73 ms |
| **MobileNet+Transformer** | 22.1M | **1.949** | **0.083** | 0.391 | 24.4% | 85 ms |

*Inference is per 16-image batch. Repetition = share of captions containing 3+ identical consecutive words.*

**The decoder dominates, not the encoder.** Swapping ResNet-18 for the 4× smaller MobileNet-v2 changes BLEU-4 by less than 0.01 in both decoder families — but swapping LSTM for Transformer moves it by more than an order of magnitude. On this dataset the bottleneck is how the caption is generated, not how the image is seen.

**The LSTM never escapes degenerate repetition.** ResNet+LSTM emits 3+ repeated words in 60% of its captions (`"a planes are are in a the with the ."`), which is why its BLEU-4 sits near zero despite a respectable METEOR of 0.22 — METEOR's unigram matching still credits the individual correct words. The Transformer's causal self-attention lets it condition on the whole prefix rather than a single hidden vector, and its repetition rate drops to 0.4%.

**MobileNet is the better value.** MobileNet+Transformer matches or beats the ResNet version on both metrics with 8.5M fewer parameters. The pooled ImageNet features are apparently more than enough for RSICD's ~1.8k-word caption vocabulary.

### Sample outputs

Same image, all four models:

```
ResNet+LSTM             many buildings are are in a of a .
MobileNet+LSTM          many buildings are in a a .
ResNet+Transformer      many cars are in a large building .
MobileNet+Transformer   many buildings are around a large playground .
```

### Dataset statistics

| Split | Images | Captions | Vocab coverage | OOV | Avg length |
|---|---|---|---|---|---|
| Train | 8,734 | 41,518 | 99.80% | 0.20% | 12.0 words |
| Validation | 1,094 | 5,069 | 98.82% | 1.18% | 10.2 words |
| Test | 1,093 | 5,129 | 98.17% | 1.83% | 12.5 words |

The vocabulary is built on **training captions only** and capped at 10k words, but a `min_word_freq` of 2 leaves just **1,803** words — RSICD's caption language is far narrower than the cap. Coverage above 98% on unseen splits means OOV is not what limits caption quality here.

---

## Experiments

<details>
<summary><b>1 — Backbone comparison: ResNet-18 vs MobileNet-v2</b></summary>

Same decoder, different encoder, measured on identical batches.

| Model | Inference | BLEU-4 | METEOR |
|---|---|---|---|
| ResNet+LSTM | 23.8 ms | 0.0036 | 0.2232 |
| MobileNet+LSTM | 22.6 ms | 0.0060 | 0.2135 |
| ResNet+Transformer | 72.7 ms | 0.0671 | 0.3922 |
| MobileNet+Transformer | 85.5 ms | 0.0677 | 0.3843 |

MobileNet costs essentially nothing in quality. The interesting cost is on the decoder side: the Transformer is 3× slower at inference because it re-encodes the growing prefix at every step, while the LSTM carries hidden state forward and does constant work per token.
</details>

<details>
<summary><b>2 — Rotation invariance</b></summary>

Overhead imagery has no canonical "up" — the same scene on a different orbit pass is the same scene rotated. Each image is captioned at 0°/90°/180°/270° and scored by mean Jaccard overlap of caption word sets against the 0° caption. Scenes are sampled across land-use keywords, because naive sampling returns airport after airport.

| Model | Consistency |
|---|---|
| ResNet+LSTM | 0.788 |
| MobileNet+Transformer | 0.749 |

Both keep roughly three-quarters of their vocabulary across rotations. Notably the LSTM scores *higher* — but it produces generic, repetitive captions that would be stable regardless of input, so a high score here partly reflects insensitivity rather than robustness.
</details>

<details>
<summary><b>3 — Vision-text interface, trained models</b></summary>

BLEU-4 on a fixed 50-sample slice, unsmoothed:

| Model | BLEU-4 |
|---|---|
| ResNet+LSTM | 0.0000 |
| MobileNet+LSTM | 0.0000 |
| ResNet+Transformer | 0.0431 |
| MobileNet+Transformer | 0.0588 |

Exact zeros for both LSTMs: with no smoothing, a caption containing no matching 4-gram scores exactly 0, and repetitive LSTM output rarely produces one.
</details>

<details>
<summary><b>4 — LSTM interface: init-hidden vs image-as-token</b></summary>

Two ways to feed the image into an LSTM — project it into the initial hidden state, or prepend it as a pseudo-word. Trained from scratch, 3 epochs, 500 training items.

| Variant | Train loss | Val loss | Convergence |
|---|---|---|---|
| Init-hidden | 1.128 | 2.636 | 2.065 |
| Image token | 1.115 | 2.660 | 2.177 |

Essentially a tie — 0.02 apart on validation. The image-token variant converges marginally faster on training loss and generalises marginally worse, consistent with it having slightly more capacity to memorise. Init-hidden is used in the main models since it costs one fewer timestep.
</details>

<details>
<summary><b>5 — Transformer memory tokens: 1 vs 4</b></summary>

How many vectors should the image expand into for cross-attention? 3 epochs, 300 training items.

| Variant | Train loss | Val loss |
|---|---|---|
| 1 token | 1.039 | 3.000 |
| 4 tokens | 0.969 | **3.075** |

4 tokens fit the training set better (−0.07) and generalise worse (+0.08). At this data scale the extra attention capacity is spent memorising, not on representing distinct aspects of the scene. The main models use 2 as a middle ground.
</details>

<details>
<summary><b>6 — Dropout placement</b></summary>

Same rate (0.3), two positions. 3 epochs, 400 training items.

| Placement | Train loss | Val loss | Train−val gap |
|---|---|---|---|
| On embeddings | 1.170 | 2.770 | −1.600 |
| In decoder block | 1.111 | 2.759 | −1.648 |

Within noise of each other. Embedding dropout produces the smaller generalisation gap, decoder dropout the marginally better absolute validation loss. Neither placement is decisive at this scale.
</details>

<details>
<summary><b>7 — Length penalty during greedy decoding</b></summary>

Greedy decoding stops early because `<eos>` is a high-frequency token. Subtracting a step-scaled amount from the `<eos>` logit after 5 tokens should delay that decision.

| Model | BLEU-4 (all penalties) | Avg length |
|---|---|---|
| ResNet+LSTM | 0.1997 | 9.4 |
| MobileNet+LSTM | 0.1783 | 8.9 |
| ResNet+Transformer | 0.2478 | 11.3 |
| MobileNet+Transformer | 0.2544 | 11.6 |

**Output was byte-identical at every penalty value 0.0 → 2.0.** This is a real result, not a broken sweep: the penalty only changes a decision when `<eos>` is close to winning the argmax, and these models are decisive enough that a gentle nudge never flips one. Meaningfully longer captions would need a hard minimum-length constraint or beam search, not a soft logit adjustment.

*(BLEU-4 here is smoothed, hence higher than the unsmoothed numbers elsewhere — the two are not comparable across sections.)*
</details>

---

## Setup

Requires Python 3.10+ and the RSICD dataset. A CUDA GPU is strongly recommended — training all four models takes roughly an hour on a P100 and is impractical on CPU.

```bash
git clone <your-repo-url>
cd rsicd-captioner

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Install PyTorch for your own CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) if the default wheel doesn't match your driver.

### Data

Download [the RSICD dataset](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset) and place the three CSVs in `data/`:

```
data/
├── train.csv
├── valid.csv
└── test.csv
```

Each CSV has `filename`, `captions`, and `image` columns, where `image` holds the raw JPEG bytes inline — there is no separate image folder to manage.

---

## Usage

```bash
# Build the vocabulary and print the text statistics table
python scripts/prepare_data.py --data-root data

# Train all four models (checkpoints land in checkpoints/)
python scripts/train.py --data-root data --epochs 5

# Train just one, with a smaller batch
python scripts/train.py --models ResNet+Transformer --batch-size 128

# Evaluate: BLEU-4, METEOR, length, repetition rate
python scripts/evaluate.py --data-root data --qualitative

# Run the ablations
python scripts/run_experiments.py --experiments all
python scripts/run_experiments.py --experiments memory_tokens dropout

# Caption a single image
python scripts/caption.py path/to/image.jpg
```

Experiments 4–6 train small purpose-built variants from scratch on a data subset, so they run without any checkpoints. Experiments 1–3 and 7 need `scripts/train.py` to have run first.

---

## Layout

```
rsicd_captioner/
├── config.py                    dataclass config for every tunable
├── data/
│   ├── tokenizer.py             word-level vocab, encode/decode/pad
│   ├── preprocessing.py         resize + ImageNet normalisation
│   ├── dataset.py               lazy-loading Dataset over the CSVs
│   └── stats.py                 coverage / OOV / length statistics
├── models/
│   ├── encoder.py               ResNet-18 / MobileNet-v2 trunk
│   ├── decoders.py              LSTM and Transformer decoders
│   └── captioners.py            assembled models + factory
├── training/
│   ├── loops.py                 AMP + gradient accumulation
│   └── memory.py                monitoring and cleanup
├── evaluation/
│   ├── metrics.py               BLEU / METEOR / repetition
│   └── evaluate.py              aggregate + qualitative + error slices
├── experiments/                 one module per ablation
└── explain/gradcam.py           Grad-CAM for captioning models

scripts/                         CLI entry points
results/metrics.json             every number reported above
main-notebook.ipynb              the original end-to-end run, outputs intact
```

`main-notebook.ipynb` is the record of the run that produced these results, with all plots and logs preserved. The package is the same pipeline refactored into importable modules.

---

## Implementation notes

Three things that were not obvious going in, and shaped the code:

**Lazy image decoding is not optional.** Decoding all ~10.9k images into memory up front costs about 6 GB — and since one image maps to five caption items, the naive version stores each image five times over. `RSICDDataset` keeps only row indices and pre-tokenised captions (tens of MB) and decodes JPEGs in `__getitem__`. This also keeps DataLoader worker processes cheap, since each one would otherwise inherit a copy.

**Mixed precision needs a dtype contract.** The dataset emits fp16 tensors to halve PCIe traffic, and training runs under `autocast`. But generation runs *outside* autocast, so inference paths explicitly cast both model and inputs to fp32 — otherwise the first matmul raises on a dtype mismatch.

**BLEU and METEOR are strict about input shape.** Both NLTK functions want lists of word strings — never index tensors, never a joined sentence. Passing a string makes NLTK iterate over characters and return a meaningless number rather than raising, so `evaluation/metrics.py` centralises the conversion.

---

## Limitations

- BLEU-4 is low in absolute terms across all models. 5 epochs on a 1.8k-word vocabulary is a baseline, not a tuned result; longer training, beam search, and spatial attention over feature maps (rather than a single pooled vector) are the obvious next steps.
- Reported metrics are on validation subsets, not the held-out test split.
- The Transformer's per-step re-encoding is the main inference cost; caching keys and values across decoding steps would remove most of it.
- Each generated caption is scored against one reference at a time, though RSICD provides five per image — scoring against all five would raise every BLEU number reported here.

## License

MIT — see [LICENSE](LICENSE).
