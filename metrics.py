import numpy as np

def classification_metrics(truth, predictions, classes):
    (y, p) = (np.asarray(truth), np.asarray(predictions))
    if y.shape != p.shape or y.size == 0:
        raise ValueError('Expected equal, nonempty truth and prediction shapes')
    if not np.isin(y, classes).all() or not np.isin(p, classes).all():
        raise ValueError('Labels outside the declared class set')
    recalls = {int(c): float(np.mean(p[y == c] == c)) if np.any(y == c) else None for c in classes}
    macro = None if any((v is None for v in recalls.values())) else float(np.mean(list(recalls.values())))
    return {'accuracy': float(np.mean(y == p)), 'recall': recalls, 'macro_recall': macro}

def detection_metrics(truth, predictions):
    result = classification_metrics(truth, predictions, [0, 1])
    return {'accuracy': result['accuracy'], 'sensitivity': result['recall'][1], 'specificity': result['recall'][0], 'balanced_accuracy': result['macro_recall']}
