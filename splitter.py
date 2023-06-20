import os
from shutil import copy
import json
import argparse


PUNC_REPLACE_DICT = {'$': ',', '@': '.', '#': '-', '_': ' '}


def silver_line_extract_text(s):
    return "".join([PUNC_REPLACE_DICT[x] if x in PUNC_REPLACE_DICT else x for x in os.path.splitext(s)[0].split("-")[-1]])


def create_datasets(seg_basenames, seg_texts, seg_dir, image_save_dir, save_dir, data_split, dataset_name):
    labeled_pairs = []
    for fn, txt in zip(seg_basenames, seg_texts):
        if "ゲ" in txt: txt = txt.replace("ゲ", "ゲ") 
        labeled_pairs.append((fn, txt))
        copy(os.path.join(seg_dir, fn), image_save_dir)
    with open(os.path.join(save_dir, f"{dataset_name}_rec_gt_{data_split}.txt"), 'w') as f:
        f.write("\n".join(["\t".join(x) for x in labeled_pairs]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--image_dir_name", type=str, required=True)
    parser.add_argument("--coco_train_name", type=str, required=True)
    parser.add_argument("--coco_val_name", type=str, required=True)
    parser.add_argument("--coco_test_name", type=str, required=True)
    parser.add_argument("--char_text_file", type=str, required=False, default=None)
    parser.add_argument("--silver_json_name", type=str, required=False, default=None)
    args = parser.parse_args()

    # set up initial dirs
    SAVE_DIR = "./train_data/rec"
    train_dir = os.path.join(SAVE_DIR, f"{args.dataset_name}_train")
    val_dir = os.path.join(SAVE_DIR, f"{args.dataset_name}_val")
    test_dir = os.path.join(SAVE_DIR, f"{args.dataset_name}_test")

    # make dirs
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # set up paths
    image_dir = os.path.join(args.root_dir, args.image_dir_name)
    coco_train_path = os.path.join(args.root_dir, args.coco_train_name)
    coco_val_path = os.path.join(args.root_dir, args.coco_val_name)
    coco_test_path = os.path.join(args.root_dir, args.coco_test_name)

    # open train and test coco jsons
    with open(coco_train_path) as f: coco_train = json.load(f)
    with open(coco_val_path) as f: coco_val = json.load(f)
    with open(coco_test_path) as f: coco_test = json.load(f)
    
    # gather texts and file names
    train_img_basenames = [x['file_name'] for x in coco_train["images"]]
    val_img_basenames = [x['file_name'] for x in coco_val["images"]]
    test_img_basenames = [x['file_name'] for x in coco_test["images"]]
    train_texts = [x['text'] for x in coco_train["images"]]
    val_texts = [x['text'] for x in coco_val["images"]]
    test_texts = [x['text'] for x in coco_test["images"]]
    if not args.silver_json_name is None:
        with open(os.path.join(args.root_dir, args.silver_json_name)) as f:
            silver_img_basenames = json.load(f)
        silver_texts = [silver_line_extract_text(x) for x in silver_img_basenames]
        train_img_basenames += silver_img_basenames
        train_texts += silver_texts
    print(f"Len val ims {len(val_img_basenames)}; len train ims \
        {len(train_img_basenames)}; len test ims {len(test_img_basenames)}")
    print(f"Len val ims {len(val_img_basenames)}; len train ims \
        {len(train_img_basenames)}; len test ims {len(test_img_basenames)}")

    # create datasets for train and test
    create_datasets(train_img_basenames, train_texts, image_dir, train_dir, SAVE_DIR, "train", args.dataset_name)
    create_datasets(val_img_basenames, val_texts, image_dir, val_dir, SAVE_DIR, "val", args.dataset_name)
    create_datasets(test_img_basenames, test_texts, image_dir, test_dir, SAVE_DIR, "test", args.dataset_name)

    # convert char text file
    if not args.char_text_file is None:
        with open(args.char_text_file) as f:
            chars = "".join(chr(int(i)) for i in f.read().split()) + " "
        with open(os.path.join(SAVE_DIR, f"{args.dataset_name}_chars.txt"), "w") as f:
            f.write("\n".join(chars))
