def split_indices(block_ids):
    values = list(block_ids)
    if not values or any((isinstance(b, bool) or int(b) != b or (not 1 <= b <= 20) for b in values)):
        raise ValueError('block_ids must be nonempty one-based integers in 1..20')
    groups = {'train': [i for (i, b) in enumerate(values) if b <= 12], 'validation': [i for (i, b) in enumerate(values) if 13 <= b <= 16], 'test': [i for (i, b) in enumerate(values) if b >= 17]}
    if not all(groups.values()):
        raise ValueError('Each split must contain samples; provide one participant with all three sets')
    return groups
