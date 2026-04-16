import numpy as np 
from tqdm import tqdm 
from datasets import load_dataset
import matplotlib.pyplot as plt 

def pil_to_float(example):
    example['image'] = np.array([example["image"]])/255 
    return example

def main(args):
    dset = load_dataset(args.dset_name)
    dset = dset.rename_column("img", "image")
    dset = dset.map(pil_to_float)
    dset.save_to_disk(args.output_dir)  

    images = np.array(dset["train"][:25]["image"])
    labels = np.array(dset["train"][:25]["label"])
    print(images.min(), images.max())
    fig, axs = plt.subplots(5, 5, figsize = (6 * 5, 6 * 5))

    for i, ax in enumerate(axs.flatten()): 
        ax.imshow(images[i].squeeze())
        ax.set(title = f"{labels[i]}")
    plt.savefig(args.output_dir + "/dset_visualization.png", bbox_inches = "tight")
if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(
        description="Fetches a hugging face dataset applies a preprocessing function to it and saves it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", required=True,
                        help="Path where the dataset will be created")
    parser.add_argument("--dset_name", required=False,
                        help="Dataset name on hugging-face, e.g. 'ylecun/mnist'")
    args = parser.parse_args()

    main(args)

