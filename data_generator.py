import numpy as np
import pandas as pd


def generate_crop_yield_data(n_samples=1000, random_state=42):
    """
    Generates a synthetic agricultural dataset similar to Kaggle's
    Crop Yield Prediction dataset.
    """
    np.random.seed(random_state)

    states = [
        "Punjab", "Haryana", "UP", "MP", "Maharashtra",
        "Karnataka", "AP", "Tamil_Nadu", "Gujarat", "Rajasthan"
    ]
    crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Pulses", "Oilseeds", "Fruits"]
    seasons = ["Kharif", "Rabi", "Zaid", "Whole Year"]

    data = {
        "State": np.random.choice(states, n_samples),
        "Crop": np.random.choice(crops, n_samples),
        "Season": np.random.choice(seasons, n_samples),
        "Area": np.random.uniform(1, 100, n_samples),                # hectares
        "Annual_Rainfall": np.random.uniform(400, 2500, n_samples),  # mm
        "Fertilizer": np.random.uniform(50, 500, n_samples),         # kg/ha
        "Pesticide": np.random.uniform(0, 50, n_samples),            # kg/ha
    }

    df = pd.DataFrame(data)

    # Yield factors
    crop_base = {
        "Rice": 3.5, "Wheat": 3.2, "Maize": 4.0, "Cotton": 1.5,
        "Sugarcane": 70.0, "Pulses": 1.2, "Oilseeds": 1.8, "Fruits": 15.0
    }
    season_factor = {"Kharif": 1.0, "Rabi": 0.9, "Zaid": 0.7, "Whole Year": 1.1}

    # Generate realistic yield (tons/ha)
    df["Yield"] = (
        df["Crop"].map(crop_base)
        + 0.001 * df["Annual_Rainfall"]
        + 0.003 * df["Fertilizer"]
        + 0.01 * df["Pesticide"]
        + 0.05 * np.log1p(df["Area"])
        + df["Season"].map(season_factor)
        + np.random.normal(0, 0.5, n_samples)
    )

    df["Yield"] = np.maximum(df["Yield"], 0.1)
    return df


if __name__ == "__main__":
    df = generate_crop_yield_data(n_samples=2000)
    df.to_csv("crop_yield.csv", index=False)
    print("Saved crop_yield.csv with shape:", df.shape)