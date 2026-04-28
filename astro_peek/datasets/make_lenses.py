import numpy as np 
from tqdm import tqdm 
from datasets import Dataset
import matplotlib.pyplot as plt
import caustics 
import torch 
from torch.nn.functional import avg_pool2d


def simulate_system(Rein, phi):
    cosmology = caustics.FlatLambdaCDM()
    cosmology.to(dtype=torch.float32)
    z_s = torch.tensor(1.0)
    z_l = torch.tensor(0.5, dtype=torch.float32)
    base_sersic = caustics.Sersic(
        x0=0.1,
        y0=0.1,
        q=0.6,
        phi=np.pi / 3,
        n=2.0,
        Re=1.0,
        Ie=1.0,
    )
    n_pix = 32
    res = 0.2
    upsample_factor = 2

    lens = caustics.SIE(
                cosmology=cosmology,
                x0=0.0,
                y0=0.0,
                q=0.6,
                phi=phi,
                Rein=Rein,
                z_l=z_l,
                z_s=z_s,
        )
    sim = caustics.LensSource(
        lens=lens,
        source=base_sersic,
        pixelscale=res,
        pixels_x=n_pix,
        upsample_factor=2,
    )
    thx, thy = caustics.utils.meshgrid(
    res / upsample_factor,
    upsample_factor * n_pix,
    dtype=torch.float32
    )

    convergence = avg_pool2d(
        lens.convergence(thx, thy).squeeze()[None, None], upsample_factor
    ).squeeze()
    return sim().numpy(), convergence

def main(args):
    # Setting the seed. 
    np.random.seed(args.seed)

    prior_params_config = {
        "einstein_radius": {"low": 0.8, "high": 1.5}, 
        "phi": {"low": -np.pi/2, "high": np.pi/4}
    }


    radius_prior = prior_params_config["einstein_radius"]
    phi_prior = prior_params_config["phi"]
    prior_sampler_radius = lambda size: np.random.uniform(radius_prior["low"], radius_prior["high"], size = size)
    prior_sampler_phi = lambda size: np.random.uniform(phi_prior["low"], phi_prior["high"], size = size)

    dataset_size = args.dataset_size

    # Creating latent factors and normalizing them...
    radii = prior_sampler_radius(dataset_size) 
    phis = prior_sampler_phi(dataset_size)
    theta = np.stack([radii, phis]).T

    # Creating simulations from it 
    images = []
    for params in tqdm(theta): 
        image = simulate_system(*params)
        images.append(image) 

    images = np.array(images)
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
        ax.set(title = r"$(R, \phi)=(%.1f, %.2f)$"%(theta[i][0], theta[i][1]))
    plt.savefig(args.output_dir + "/dset_visualization.png", bbox_inches = "tight")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create strong lensing systems dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", required=True,
                        help="Path where the dataset will be created")
    parser.add_argument("--img_size", required=False, default = 32, type = int,
                        help="Number of rows/cols in the image")
    parser.add_argument("--dataset_size", required = False, default = 10_000, type = int)
    parser.add_argument("--seed", required = False, default = 42, type = int)

    args = parser.parse_args()
    main(args)
