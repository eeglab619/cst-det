# Data interface

One numeric NPZ file per participant, loaded with `allow_pickle=False`.
This is the public framework interface; data conversion is withheld.

`eeg`: `[N, 8, 1536]`; `state_labels`: `[N, 48]`, NC=0/IC=1;
`block_ids`: `[N]`, one-based IDs 1–20. Windows must not cross blocks.

Channel order: P3, PO7, O1, Oz, Pz, P4, PO8, O2. Sampling rate: 256 Hz.
Apply the paper's 0.14 s temporal shift. Preserve source block identities.
Raw recordings and checkpoints are not included.
