import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import tqdm


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
        default=[2, 256],
        help="List of sizes for normalization",
    )
    return parser.parse_args()


def main(args):
    Input_path = args.Input_path
    sizes = args.size

    Image_names = os.listdir(Input_path)  # 获取目录下所有图片名称列表
    print(len(Image_names))

    colors = ["r", "g", "c", "b"]
    labels = ["iCT", "Optima", "Revolution", "SOMATOM"]

    for size in sizes:
        data = np.zeros(
            (len(Image_names), size * size)
        )  # 初始化一个np.array数组用于存数据
        label = np.zeros((len(Image_names),))  # 初始化一个np.array数组用于存标签

        # 读取并存储图片数据,原图为rgb三通道,而且大小不一,先灰度化,再resize成指定大小
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

        tsne_2D = TSNE(
            n_components=2, init="pca", perplexity=2, random_state=0
        )  # 调用TSNE
        result_2D = tsne_2D.fit_transform(data)

        x_min, x_max = np.min(result_2D, 0), np.max(result_2D, 0)
        embedding = (result_2D - x_min) / (x_max - x_min)

        plt.figure(figsize=(10, 10))

        # 遍历每个数据点,根据标签绘制不同颜色的点
        for i in range(embedding.shape[0]):
            plt.scatter(
                embedding[i, 0],
                embedding[i, 1],
                c=colors[int(label[i])],
                s=80,
                marker=".",
            )

        legend_elements = [
            plt.Line2D([0], [0], color=c, marker=".", linestyle="") for c in colors
        ]
        plt.legend(legend_elements, labels, loc="upper right")

        plt.xticks([])
        plt.yticks([])
        plt.title(f"t-SNE Visualization (size={size})")
        plt.tight_layout()

        plt.savefig(f"t_sne_OD_size_{size}.png", dpi=200)  # 保存图像到当前路径
        plt.close()  # 关闭图像,节省内存


if __name__ == "__main__":
    args = parse_args()
    main(args)
