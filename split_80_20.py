
import os, random, shutil

# paths
base_dir = "/home/mert/X-ray-ai/dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir,"val")

# ensure validation folders exits
os.makedirs(os.path.join(val_dir,"normal"), exists_ok = True)
os.makedirs(os.path.join(val_dir, "osteoarthritis"), exists_ok = True)

val_ratio = 0.2

for category in ["normal", "osteoarthritis"]:
    train_path = os.path.join(train_dir, category)
    val_path = os.path.join(val_dir, category)

    # list all images in this category and shuffle them
    images = os.listdir(train_path)
    random.shuffle(images)

    # number of validation images
    val_count = int(len(images) * val_ratio)

    # move them to validation folder
    for img in images[:val_count]:
        shutil.move(os.path.join(train_path, img), os.path.join(val_path, img))

    print(f"moved {val_count} images to validation.")


    
