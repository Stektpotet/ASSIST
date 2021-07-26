import os
import yaml
from shutil import copyfile

dataset_folder = "../data/catchsnap/raw"
split_file = "../data/catchsnap/dataset_split.yaml"
split_folder = "../data/catchsnap/split"

# if __name__ == '__main__':
with open(split_file, "r") as file:
    split_dict = yaml.load(file, Loader=yaml.FullLoader)

os.makedirs(split_folder, exist_ok=True)

train_folder = os.path.join(split_folder, "train")
os.makedirs(train_folder, exist_ok=True)
for i, train_file in enumerate(split_dict["train_files"]):
    parent_folder = train_file.split("\\")[0]
    if not os.path.isdir(os.path.join(train_folder, parent_folder)):
        os.makedirs(os.path.join(train_folder, parent_folder), exist_ok=True)
    copyfile(os.path.join(dataset_folder, train_file), os.path.join(train_folder, train_file))
    print(f"Copying train data [{i + 1}/{len(split_dict['train_files'])}]")

test_folder = os.path.join(split_folder, "test")
os.makedirs(train_folder, exist_ok=True)
for i, test_file in enumerate(split_dict["test_files"]):
    parent_folder = test_file.split("\\")[0]
    if not os.path.isdir(os.path.join(test_folder, parent_folder)):
        os.makedirs(os.path.join(test_folder, parent_folder), exist_ok=True)
    copyfile(os.path.join(dataset_folder, test_file), os.path.join(test_folder, test_file))
    print(f"Copying test data [{i + 1}/{len(split_dict['test_files'])}]")