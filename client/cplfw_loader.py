import os

def load_cplfw(dataset_dir, pairs_file):
    """
    Loads CPLFW dataset verification pairs.

    The pairs file contains two consecutive lines per pair:
    the first line includes the first image and label (1=same, 0=different),
    the second line contains the second image filename.

    Parameters
    ----------
    dataset_dir : str
        Folder containing CPLFW aligned images.
    pairs_file : str
        Text file specifying the verification pairs.

    Returns
    -------
    list of tuples
        List of (image1_path, image2_path, label) where label is
        1 for same person, 0 for different persons.
    """
    pairs = []

    with open(pairs_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for i in range(0, len(lines), 2):
        line1 = lines[i].split()
        line2 = lines[i + 1].split()

        img1_name = line1[0]
        label = int(line1[1])  # 1 or 0

        img2_name = line2[0]

        img1 = os.path.join(dataset_dir, img1_name)
        img2 = os.path.join(dataset_dir, img2_name)

        pairs.append((img1, img2, label))

    return pairs
