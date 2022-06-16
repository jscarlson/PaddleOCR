import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.dict_file) as f:
        chars_dec= f.read().split()
        chars = [chr(c) for c in chars_dec]

    print(f"Len of char set: {len(chars)}")

    with open(args.dict_file.replace(".txt", "_converted.txt"), "w") as f:
        f.write("\n".join(chars))
