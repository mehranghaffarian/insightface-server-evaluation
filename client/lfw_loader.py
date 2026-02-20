import os


def load_lfw(dataset_dir, pairs_file):
    """
    Loads LFW dataset verification pairs.

    Each line in the pairs file defines either a same-person pair
    (3 columns) or a different-person pair (4 columns).

    Parameters
    ----------
    dataset_dir : str
        Root folder containing LFW images organized by person.
    pairs_file : str
        CSV file specifying the verification pairs.

    Returns
    -------
    list of tuples
        List of (image1_path, image2_path, label) where label is
        1 for same person, 0 for different persons.
    """
    pairs = []


    with open(pairs_file, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(",")
        parts = [p for p in parts if p != ""]

        if len(parts) == 3:
            # Same person
            name, idx1, idx2 = parts

            img1 = os.path.join(
                dataset_dir,
                name,
                f"{name}_{int(idx1):04d}.jpg"
            )

            img2 = os.path.join(
                dataset_dir,
                name,
                f"{name}_{int(idx2):04d}.jpg"
            )

            label = 1

        elif len(parts) == 4:
            # Different people
            name1, idx1, name2, idx2 = parts

            img1 = os.path.join(
                dataset_dir,
                name1,
                f"{name1}_{int(idx1):04d}.jpg"
            )

            img2 = os.path.join(
                dataset_dir,
                name2,
                f"{name2}_{int(idx2):04d}.jpg"
            )

            label = 0

        else:
            continue

        pairs.append((img1, img2, label))

    return pairs
