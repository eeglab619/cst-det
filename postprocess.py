from release import RELEASE_NOTICE

def fill_gaps(sequence, gap_limit_points):
    raise NotImplementedError(RELEASE_NOTICE)

def remove_noise(sequence, noise_limit_points):
    raise NotImplementedError(RELEASE_NOTICE)

def temporal_shaping(sequence, gap_limit_points=5, noise_limit_points=4):
    return remove_noise(fill_gaps(sequence, gap_limit_points), noise_limit_points)
