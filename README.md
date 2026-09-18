# CST-Det

Code framework for the ICASSP 2027 submission **CST-Det: A High-Temporal-Resolution State Detection Framework for Continuous SSVEP Decoding**.

> Core implementations are withheld during peer review and raise
> `NotImplementedError`. The full implementation will be released in this
> repository after the paper is accepted. This partial release cannot
> reproduce the reported results.

## Framework

```text
EEG -> spatiotemporal features -> learnable smoothing -> sigmoid
    -> threshold -> temporal shaping -> IC intervals -> FBCCA
```

`models/` contains the model interfaces, `pipeline.py` the experiment flow,
`config.json` the paper settings, and `protocol.py` the block split.
Core model, training and data preparation details remain withheld.
Unspecified architecture dimensions and training choices are not filled in
with legacy defaults. `models/fbcca.py` is a supporting baseline utility.

## Protocol

16 participants; 8 targets (8–15 Hz); durations 0.5–2.5 s; 256 Hz EEG;
0.14 s label/segment shift. Within each participant, blocks 1–12 train,
13–16 validate, and 17–20 test. Training uses 100 epochs and learning rate 0.001.

6 s inputs; 0.125 s output interval; Adam; binary cross-entropy;
batch size 32; threshold 0.5; smoothing kernel 5; closing/opening lengths 5/4.
Smoothing uses preceding state scores only. Temporal shaping applies closing
then opening. The downstream classifier is fixed FBCCA.

## Usage

```bash
python run.py --describe
python -m pip install -r requirements.txt
python run.py --data data/subject_01.npz
```

The final command intentionally raises `NotImplementedError` until the full
release. Data and checkpoints are not included. The prepared-data interface
is listed in [data/README.md](data/README.md).
