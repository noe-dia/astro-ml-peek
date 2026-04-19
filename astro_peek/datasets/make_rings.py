import numpy as np 
from tqdm import tqdm 
from datasets import Dataset
import matplotlib.pyplot as plt

def make_rings(x, y, radius = 10, sigma = 3, center = 32//2): 
    """
    x and y are arrays created from a np.meshgrid. 
    """
    r =  np.sqrt((x-center) ** 2 + (y - center) ** 2)
    return np.exp( - (r - radius) ** 2 / sigma)


def main(args):
    # Setting the seed. 
    np.random.seed(args.seed)
    # Setting up the hyperparameters for the ring  
    img_size = args.img_size
    x = np.arange(0, img_size) / img_size
    y = np.arange(0, img_size) / img_size
    X, Y = np.meshgrid(x,y)


    prior_params_config = {
        "radius": {"low": 0.1, "high": 0.3}, 
        "sigma": {"low": 1e-4, "high": 1e-2}
    }

    radius_prior = prior_params_config["radius"]
    sigma_prior = prior_params_config["sigma"]
    prior_sampler_radius = lambda size: np.random.uniform(radius_prior["low"], radius_prior["high"], size = size)
    prior_sampler_sigma = lambda size: np.random.uniform(sigma_prior["low"], sigma_prior["high"], size = size)

    dataset_size = args.dataset_size

    # Creating latent factors and normalizing them...
    radius = prior_sampler_radius(size = dataset_size)
    sigma =  prior_sampler_sigma(size = dataset_size)
    theta = np.stack([radius, sigma]).T

    # Creating simulations from it 
    images = []
    for params in tqdm(theta): 
        image = make_rings(X, Y, radius = params[0], sigma = params[1], center = 0.5)
        images.append(image) 

    images = np.array(images)
    radius = (radius - radius_prior["low"]) / (radius_prior["high"] - radius_prior["low"])
    sigma =  (sigma - sigma_prior["low"]) / (sigma_prior["high"] - sigma_prior["low"])
    theta = np.stack([radius, sigma]).T
    dset = Dataset.from_dict({
                "theta": theta,
                "image": images 
            })
    dset = dset.train_test_split(train_size = 0.8, test_size=0.2, seed=args.seed)
    dset.save_to_disk(args.output_dir)


    # Plotting a few rings as a sanity check
    fig, axs = plt.subplots(5, 5, figsize = (6 * 5, 6 * 5))

    for i, ax in enumerate(axs.flatten()): 
        ax.imshow(images[i].squeeze())
        ax.set(title = r"$(R, \sigma)=(%.2f, %.2f)$"%(theta[i][0], theta[i][1]))
    plt.savefig(args.output_dir + "/dset_visualization.png", bbox_inches = "tight")



if __name__ == "__main__":
    import argparse

    from datasets import load_from_disk

    parser = argparse.ArgumentParser(
        description="Create ring dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", required=True,
                        help="Path where the dataset will be created")
    parser.add_argument("--img_size", required=False, default = 32, type = int,
                        help="Number of rows/cols in the image")
    parser.add_argument("--radius_prior", required = False, default = [0.1, 0.43])
    parser.add_argument("--sigma_prior", required = False, default = [0.1, 0.43])
    parser.add_argument("--dataset_size", required = False, default = 10_000, type = int)
    parser.add_argument("--seed", required = False, default = 42, type = int)

    args = parser.parse_args()
    main(args)
