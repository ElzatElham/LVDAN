import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import tqdm
from sklearn.decomposition import PCA
from umap import UMAP

def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE Visualization")
    parser.add_argument(
        "--Input_path",
        type=str,
        help="Path to the input image folder",
        default="/Volumes/ZelinDisk/big/YJT_DA/whole_images/",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        default=[96],
        help="List of sizes for normalization",
    )
    return parser.parse_args()


def main(args):
    Input_path = args.Input_path
    sizes = args.size

    Image_names = os.listdir(Input_path)
    print(len(Image_names))

    colors = ["r", "g", "c", "b"]
    labels = ["iCT", "Optima", "Revolution", "SOMATOM"]

    for size in sizes:
        data = np.zeros(
            (len(Image_names), size * size)
        )
        label = np.zeros((len(Image_names),))

        for i in tqdm.tqdm(range(len(Image_names))):
            image_path = os.path.join(Input_path, Image_names[i])
            basename = os.path.basename(image_path)

            if "iCT" in basename:
                label[i] = 0
            elif "Revolution" in basename:
                label[i] = 1
            elif "Optima" in basename:
                label[i] = 2
            elif "SOMATOM" in basename:
                label[i] = 3

            img = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img_gray, (size, size))
            img = img.reshape(1, size * size)
            data[i] = img
            
        # save data and label to .tem file
        
        name = f"./.tem/data_size_{size}.npy"
        if os.path.exists(name):
            data = np.load(name)
            label = np.load(f"./.tem/label_size_{size}.npy")
        else:
            np.save(f"./.tem/data_size_{size}.npy", data)
            np.save(f"./.tem/label_size_{size}.npy", label)

        if data.shape[1] > 100:
            # pca decomposition
            data = PCA(n_components=100).fit_transform(data)
            

        downsample = 5000
        downsample_index = np.random.choice(
            range(len(data)), downsample, replace=False
        )
        data = data[downsample_index]
        label = label[downsample_index]

        # drer = TSNE(
        #     n_components=2, init="pca", perplexity=30, random_state=0
        # )
        drer = UMAP(n_components=2, random_state=0)
        result_2D = drer.fit_transform(data)

        x_min, x_max = np.min(result_2D, 0), np.max(result_2D, 0)
        embedding = (result_2D - x_min) / (x_max - x_min)

        plt.figure(figsize=(10, 10))

        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=label,
            s=80,
            marker=".",
        )

        plt.xticks([])
        plt.yticks([])
        plt.title(f"t-SNE Visualization (size={size})")
        plt.tight_layout()

        plt.savefig(f"t_sne_OD_size_{size}.png", dpi=200)  # 保存图像到当前路径
        plt.close()  # 关闭图像,节省内存


if __name__ == "__main__":
    args = parse_args()
    main(args)
