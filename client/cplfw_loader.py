import os

def load_cplfw(dataset_dir, pairs_file):
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
