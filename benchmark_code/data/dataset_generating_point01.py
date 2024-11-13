"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from benchpots.datasets import preprocess_ett
from pypots.utils.random import set_random_seed

from utils import organize_and_save

if __name__ == "__main__":
    set_random_seed(2024)
    rate = 0.1
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

    set_random_seed(2024)
    rate = 0.2
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

    set_random_seed(2024)
    rate = 0.4
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

    set_random_seed(2024)
    rate = 0.6
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


    set_random_seed(2024)
    rate = 0.8
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

    set_random_seed(2024)
    rate = 0.9
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

