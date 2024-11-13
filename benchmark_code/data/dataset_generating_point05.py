"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from benchpots.datasets import (
    preprocess_beijing_air_quality,
    preprocess_italy_air_quality,
    preprocess_electricity_load_diagrams,
    preprocess_ett,
    preprocess_pems_traffic,
    preprocess_ucr_uea_datasets,
)
from pypots.utils.random import set_random_seed

from utils import organize_and_save

if __name__ == "__main__":
    set_random_seed(2024)
    rate = 0.5
    pattern = "point"

    step = 48
    ett = preprocess_ett(
        subset="ETTh1",
        rate=rate,
        n_steps=step,
        pattern=pattern,
    )
    organize_and_save(
        ett, f"generated_datasets/ett_rate{int(rate * 10):02d}_step{step}_{pattern}"
    )
