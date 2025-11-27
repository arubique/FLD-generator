import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-out", type=str, default=None)
    parser.add_argument("--val-out", type=str, default=None)
    parser.add_argument("--test-out", type=str, default=None)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    with open(args.input_path, "r") as f:
        lines = f.readlines()

    random.seed(args.seed)
    random.shuffle(lines)

    n = len(lines)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_lines = lines[:n_train]
    val_lines = lines[n_train:n_train + n_val]
    test_lines = lines[n_train + n_val:]

    base = args.input_path
    train_path = args.train_out or (base + ".train")
    val_path = args.val_out or (base + ".val")
    test_path = args.test_out or (base + ".test")

    with open(train_path, "w") as f:
        f.writelines(train_lines)
    with open(val_path, "w") as f:
        f.writelines(val_lines)
    with open(test_path, "w") as f:
        f.writelines(test_lines)

    print(f"train: {len(train_lines)} -> {train_path}")
    print(f"val:   {len(val_lines)} -> {val_path}")
    print(f"test:  {len(test_lines)} -> {test_path}")

if __name__ == "__main__":
    main()
