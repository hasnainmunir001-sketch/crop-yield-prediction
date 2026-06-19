import argparse
from data_generator import generate_crop_yield_data
from pipeline import CropYieldPipeline


def main():
    parser = argparse.ArgumentParser(description="Train Crop Yield Prediction models")
    parser.add_argument("--samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--save", action="store_true", help="Save trained models")
    args = parser.parse_args()

    print("Generating dataset...")
    df = generate_crop_yield_data(n_samples=args.samples)

    print("Initializing pipeline...")
    pipe = CropYieldPipeline(df)
    pipe.split_data()

    print("\n1. Supervised Learning")
    print(pipe.train_supervised())

    print("\n2. Neural Network")
    print(pipe.train_neural_network())

    print("\n3. Semi-Supervised Learning")
    print(pipe.train_semi_supervised(labeled_ratio=0.2))

    print("\n4. Unsupervised Clustering")
    print(pipe.get_unsupervised_clusters().head())

    if args.save:
        pipe.save_models()
        print("\nModels saved successfully!")


if __name__ == "__main__":
    main()