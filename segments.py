import numpy as np

def find_events(mask, scale_factor=32):
    mask = mask.astype(int)
    if mask.ndim == 2:
        batch_events_list = []
        for i in range(mask.shape[0]):
            events = find_events(mask[i], scale_factor)
            batch_events_list.append(events)
        return batch_events_list
    padded_mask = np.pad(mask, (1, 1), 'constant', constant_values=0)
    diff = np.diff(padded_mask)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    signal_starts = starts * scale_factor
    signal_ends = ends * scale_factor
    event_ranges = np.column_stack((signal_starts, signal_ends))
    return event_ranges

def slice_signal_by_events(signal_data, events):
    all_batches_segments = []
    if signal_data.ndim == 3 and isinstance(events, list):
        assert len(events) == signal_data.shape[0], 'Mask批次数量与Signal批次数量不一致'
        for b in range(signal_data.shape[0]):
            curr_signal = signal_data[b]
            curr_events = events[b]
            current_batch_segments = []
            for (start, end) in curr_events:
                segment = curr_signal[:, start:end]
                current_batch_segments.append(segment)
            all_batches_segments.append(current_batch_segments)
    else:
        if isinstance(events, list) and len(events) > 0 and (not isinstance(events[0], (int, float))):
            events = events[0]
        current_batch_segments = []
        for (start, end) in events:
            segment = signal_data[:, start:end]
            current_batch_segments.append(segment)
        all_batches_segments.append(current_batch_segments)
    return all_batches_segments
