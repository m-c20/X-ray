import os
import cv2
from matplotlib import pyplot as plt

# paths to dataset folders




train_dir = "dataset/train"
normal_dir = os.path.join(train_dir, "n")
oa_dir = os.path.join(train_dir, "oa")

# Functions to display few images from folder

def show_images(folder, num_images=3):

    images = os.listdir(folder)[:num_images]
    for img_name in images:
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(img_name)
        plt.axis("off")
        plt.savefig(f"test_the_{img_name}.png")
        plt.close()




print("Normal X-rays:")
show_images(normal_dir)

print("OA X-rays:")
show_images(oa_dir)

