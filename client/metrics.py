import numpy as np

def compute_metrics(similarities, labels, threshold):
    """
    Computes verification metrics for a set of predictions.

    Parameters
    ----------
    similarities : array-like
        Cosine similarity scores between image pairs.
    labels : array-like
        Ground-truth labels (1 = same person, 0 = different persons).
    threshold : float
        Similarity threshold for classifying a match.

    Returns
    -------
    tuple of floats
        (accuracy, FMR, FNMR)
        - accuracy : overall verification accuracy
        - FMR : False Match Rate
        - FNMR : False Non-Match Rate
    """
    similarities = np.array(similarities)
    labels = np.array(labels)

    predictions = (similarities >= threshold).astype(int)

    TP = np.sum((predictions == 1) & (labels == 1))
    TN = np.sum((predictions == 0) & (labels == 0))
    FP = np.sum((predictions == 1) & (labels == 0))
    FN = np.sum((predictions == 0) & (labels == 1))

    accuracy = (TP + TN) / len(labels)

    FMR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNMR = FN / (FN + TP) if (FN + TP) > 0 else 0

    return accuracy, FMR, FNMR
