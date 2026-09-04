# generate_train_txt.py

def generate_train_ids(output_file="test_design_ids.txt", total=550):
    """
    Generate first 440 features_XXXXX IDs
    and write them one per line into a txt file.
    """

    with open(output_file, "w") as f:
        for i in range(496, total + 1):
            f.write(f"features_{i:05d}\n")

    print(f"Generated {total} IDs in {output_file}")


if __name__ == "__main__":
    generate_train_ids()

