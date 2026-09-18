import numpy as np
from release import require_core
from protocol import split_indices
from metrics import detection_metrics

def run_experiment(path, config):
    require_core()
    from models.cst_det import CSTDet
    from engine import train_model
    from postprocess import temporal_shaping
    model = CSTDet()
    with np.load(path, allow_pickle=False) as data:
        (x, y, blocks) = (data['eeg'], data['state_labels'], data['block_ids'])
    if x.ndim != 3 or x.shape[1:] != (8, 1536) or y.shape != (len(x), 48) or (blocks.shape != (len(x),)):
        raise ValueError('Expected eeg [N,8,1536], state_labels [N,48], block_ids [N]')
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid EEG or binary labels')
    split = split_indices(blocks)
    (_, probabilities) = train_model(model, x[:, None], y, split, config)
    states = np.asarray([temporal_shaping(row >= config['threshold'], config['closing_points'], config['opening_points']) for row in probabilities])
    return detection_metrics(y[split['test']], states)

def decode_candidate_segments(eeg, states, config):
    require_core()
    from segments import find_events
    from models.fbcca import FBCCA
    decoder = FBCCA(config['target_frequencies_hz'], fs=config['sample_rate'], max_duration_sec=eeg.shape[-1] / config['sample_rate'])
    predictions = np.zeros(len(states), dtype=int)
    for (start, end) in find_events(np.asarray(states), config['stride_samples']):
        (_, target, _) = decoder.predict(eeg[:, start:end])
        predictions[start // config['stride_samples']:end // config['stride_samples']] = target + 1
    return predictions
