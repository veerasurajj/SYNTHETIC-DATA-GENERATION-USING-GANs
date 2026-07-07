# Synthetic Industrial Sensor Data Generator (CTGAN + WGAN)

## What this project does

Factories collect sensor data (temperature, pressure, vibration, etc.) from
machines to predict failures before they happen. But real failure data is
often scarce, sensitive, or hard to share.

This project **generates realistic fake (synthetic) industrial sensor data**
that looks statistically similar to real data, without exposing any real
machine records. It does this using two different generative AI models and
then combines their outputs into one richer dataset.

Think of it like a data "photocopier" that doesn't copy the original data
point-for-point, but instead learns the *patterns* in the data (how
temperature, pressure, vibration, and failures tend to relate to each other)
and produces brand-new, realistic-looking rows based on those patterns.

## Why this is useful

- **Privacy-safe testing**: Share/test data without exposing real machine data.
- **Data augmentation**: Get more training examples for machine-learning
  models when real data is limited.
- **Rare-event simulation**: Generate more examples of rare situations
  (e.g., machine failures) that don't happen often in real life.

## How it works (step by step)

1. **Load / simulate the data**
   The notebook starts with an example industrial dataset of 1,000 machine
   readings (in your own use, you'd replace this with your real dataset).

2. **Normalize the data**
   All values are scaled to a 0–1 range so the models can learn efficiently.

3. **Train Model 1: CTGAN**
   [CTGAN](https://github.com/sdv-dev/CTGAN) is a generative model built
   specifically for tabular (spreadsheet-like) data. It learns the patterns
   in the scaled dataset and generates 500 new synthetic rows.

4. **Train Model 2: WGAN (Wasserstein GAN)**
   A second, custom-built generative model (a Generator + Discriminator pair)
   is trained from scratch on the same data. It also learns to produce
   realistic synthetic rows using an adversarial training process (the
   Generator tries to fool the Discriminator into thinking its fake data is
   real).

5. **Combine into a hybrid dataset**
   The synthetic outputs from CTGAN and the WGAN are merged together into one
   combined synthetic dataset, then converted back to the original
   (un-normalized) scale.

6. **Visualize & compare**
   Histograms and a scatter plot compare the original data against the
   synthetic data, so you can visually check how closely the synthetic data
   matches the real data's patterns.

## Inputs

| Input | Description |
|---|---|
| **Dataset** | A table with columns: `temperature`, `pressure`, `vibration`, `machine_age`, `failure`. In the notebook this is randomly simulated, but you can swap in your own real sensor data (same column names/format). |
| **Training settings** | Number of training rounds ("epochs") for CTGAN (default 50) and for the WGAN (default 50). More epochs generally means better-quality synthetic data but longer training time. |
| **Number of synthetic samples** | How many new synthetic rows to generate (default 500 from each model). |

## Outputs

| Output | Description |
|---|---|
| `synthetic_ctgan` | 500 synthetic rows produced by the CTGAN model. |
| `synthetic_wgan` | 500 synthetic rows produced by the custom WGAN model. |
| `synthetic_combined` | The final hybrid dataset — CTGAN and WGAN outputs merged together and rescaled back to real-world units (e.g., actual temperature values, not 0–1 numbers). This is the main deliverable. |
| **Comparison charts** | Histograms of each variable (temperature, pressure, vibration, machine age) showing original data (blue) vs. synthetic data (orange), plus a scatter plot comparing temperature vs. pressure for both datasets. These let you visually judge how realistic the synthetic data is. |

## The data columns explained

| Column | Meaning |
|---|---|
| `temperature` | Machine operating temperature (°, simulated range 20–100) |
| `pressure` | Machine pressure reading (simulated range 1–10) |
| `vibration` | Machine vibration level (simulated range 0.1–5.0) |
| `machine_age` | Age of the machine in years (1–10) |
| `failure` | Whether the machine failed (`1`) or not (`0`) — this is a category, not a continuous number |

## How to run it

1. Install the required libraries (first cell of the notebook):
   ```
   pip install pandas numpy matplotlib seaborn torch ctgan scikit-learn
   ```
2. Run the notebook cells in order, top to bottom.
3. At the end, the `synthetic_combined` dataframe holds your final synthetic
   dataset. Uncomment the line in the notebook to save it as a CSV file:
   ```python
   synthetic_combined.to_csv('synthetic_industrial_data.csv', index=False)
   ```

## Things to keep in mind

- The dataset in this notebook is **randomly simulated** for demonstration.
  Swap it out with your real dataset (same column names) to generate
  synthetic versions of your actual data.
- More training epochs generally improve realism but take longer to run.
- Always visually/statistically check synthetic data (using the provided
  charts) before relying on it, since generative models can sometimes miss
  subtle real-world patterns.
