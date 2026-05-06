from astro_peek.utils import load_yaml 
import numpy as np 
from datasets import Dataset, DatasetInfo, concatenate_datasets
from astro_peek.datasets import *
from tqdm import tqdm

def str_to_float_cfg(cfg, keys = None):
    for key in keys:
        cfg[key] = float(cfg[key])
    return cfg

def main(args): 
    # Fixing the seed
    np.random.seed(args.seed_init)

    print("Loading configs and instantiating priors...")
    cfg = load_yaml("../../astro_peek/datasets/cosmo_prior.yaml")
    prior_cfg = cfg['priors']
    pk_cfg = str_to_float_cfg(cfg["matter_pk"], keys = ["kmin", "kmax"])
    delta_cfg = str_to_float_cfg(cfg["density_field"], keys = ["kmin", "kmax"])
    prior = instantiate_prior(prior_cfg)

    print("Running main loop to create LogNormal fields.")
    num_cosmo = args.num_cosmo
    psis = sample_prior(prior, num_cosmo)
    seed = args.seed_init
    num_fields_per_cosmo = args.num_fields_per_cosmo
    img_size = int(delta_cfg['img_size'])
    deltas = np.empty(shape = (num_cosmo, num_fields_per_cosmo, img_size, img_size))   # (N_cosmo, N_field_per_cosmo, img_size, img_size)
    pks =   np.empty(shape = (num_cosmo, 300)) # (N_cosmo, 300)
    dsets = []
    # Looping over cosmologies to create power spectrum and density contrast fields.
    for i in tqdm(range(num_cosmo)): 
        psi = psis[i]
        prior_sample = {key: psi[i] for i, key in enumerate(prior.keys())}
        k, Pk = compute_pk(prior_sample, **pk_cfg)
        pks[i] = Pk

        for j in range(num_fields_per_cosmo):
            delta = compute_density_contrast_from_scratch(prior_sample, seed = seed, return_volume = False, **delta_cfg)
            deltas[i, j] = delta
            seed += 1

        dsets.append(
            Dataset.from_dict({
                "power_spectrum": pks,
                "density_contrast": deltas ,
                "cosmo_params": psis})
            )
        info = DatasetInfo(description = "Dataset of matter power spectra and Lognormal density contrasts for varying (H0, Omega_b, Omega_c, n_s, A_s)")
    
    print("Concatenating datasets...")
    dset = concatenate_datasets(dsets, info = info)

    print("Saving everything to disk...")
    dset.save_to_disk(args.output_dir)
    



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create strong lensing systems dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", required=True, help="Path where the dataset will be created")
    parser.add_argument("--num_cosmo", required = False, default = 1_000, type = int)
    parser.add_argument("--num_fields_per_cosmo", required = False, default = 10, type = int)
    parser.add_argument("--seed_init", required = False, default = 0, type = int)

    args = parser.parse_args()
    main(args)