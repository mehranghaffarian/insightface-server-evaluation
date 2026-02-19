import os


def load_lfw(dataset_dir, pairs_file):
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
