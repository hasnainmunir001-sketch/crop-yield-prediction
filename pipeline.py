import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def create_preprocessor(X):
    """Build a fresh ColumnTransformer for categorical + numerical features."""
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = X.select_dtypes(include=[np.number]).columns.tolist()

    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_transformer = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical),
            ("cat", cat_transformer, categorical),
        ]
    )


class SemiSupervisedModel:
    """Wrapper for self-training regression model."""
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, X):
        X_proc = self.preprocessor.transform(X)
        return self.model.predict(X_proc)


class CropYieldPipeline:
    def __init__(self, df):
        self.df = df.copy()
        self.target = "Yield"
        self.features = [c for c in df.columns if c != self.target]
        self.models = {}
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def split_data(self, test_size=0.2):
        X = self.df[self.features]
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    def train_supervised(self):
        """Train Random Forest & XGBoost (supervised)."""
        results = []
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        }

        for name, model in models.items():
            pipe = Pipeline([
                ("preprocessor", create_preprocessor(self.X_train)),
                ("selector", SelectKBest(f_regression, k="all")),
                ("model", model),
            ])
            pipe.fit(self.X_train, self.y_train)
            self.models[name] = pipe

            y_pred = pipe.predict(self.X_test)
            results.append(self._score(name, y_pred))

        return pd.DataFrame(results)

    def train_neural_network(self):
        """Train an MLP Neural Network."""
        pipe = Pipeline([
            ("preprocessor", create_preprocessor(self.X_train)),
            ("selector", SelectKBest(f_regression, k="all")),
            ("model", MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)),
        ])
        pipe.fit(self.X_train, self.y_train)
        self.models["Neural Network (MLP)"] = pipe

        y_pred = pipe.predict(self.X_test)
        return pd.DataFrame([self._score("Neural Network (MLP)", y_pred)])

    def train_semi_supervised(self, labeled_ratio=0.2):
        """
        Semi-supervised learning via self-training.
        Uses only a small labeled portion and pseudo-labels the rest.
        """
        n_labeled = int(len(self.y_train) * labeled_ratio)
        X_labeled = self.X_train.iloc[:n_labeled]
        y_labeled = self.y_train.iloc[:n_labeled]
        X_unlabeled = self.X_train.iloc[n_labeled:]

        preprocessor = create_preprocessor(self.X_train)
        X_lab_proc = preprocessor.fit_transform(X_labeled)
        X_unlab_proc = preprocessor.transform(X_unlabeled)

        # Initial model on labeled data
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_lab_proc, y_labeled)

        # Pseudo-label unlabeled data
        y_pseudo = model.predict(X_unlab_proc)

        # Combine and retrain
        X_combined = np.vstack([X_lab_proc, X_unlab_proc])
        y_combined = np.concatenate([y_labeled.values, y_pseudo])

        final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        final_model.fit(X_combined, y_combined)

        self.models["Semi-Supervised (Self-Training)"] = SemiSupervisedModel(preprocessor, final_model)

        y_pred = final_model.predict(preprocessor.transform(self.X_test))
        return pd.DataFrame([{
            **self._score("Semi-Supervised (Self-Training)", y_pred),
            "Labeled Samples": n_labeled,
            "Unlabeled Samples": len(X_unlabeled),
        }])

    def get_unsupervised_clusters(self, n_clusters=4):
        """
        Unsupervised K-Means clustering on preprocessed features + PCA for 2D viz.
        """
        preprocessor = create_preprocessor(self.X_train)
        X_proc = preprocessor.fit_transform(self.X_train)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_proc)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_proc)

        return pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": clusters,
            "Yield": self.y_train.values,
        })

    def get_feature_importance(self, model_name="Random Forest"):
        """Get feature importances from tree-based models."""
        if model_name not in self.models:
            return None

        model = self.models[model_name]
        if not hasattr(model, "named_steps"):
            return None

        preprocessor = model.named_steps["preprocessor"]
        selector = model.named_steps["selector"]
        tree_model = model.named_steps["model"]

        # Get feature names after preprocessing
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feat_{i}" for i in range(preprocessor.transform(self.X_train).shape[1])]

        # Apply selector mask
        mask = selector.get_support()
        feature_names = np.array(feature_names)[mask]

        importances = tree_model.feature_importances_
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False)

    def cross_validate(self, model_name, cv=5):
        """Perform cross-validation on a trained model."""
        model = self.models[model_name]
        if hasattr(model, "named_steps"):
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring="r2", n_jobs=-1)
        else:
            preprocessor = model.preprocessor
            X_proc = preprocessor.transform(self.X_train)
            scores = cross_val_score(model.model, X_proc, self.y_train, cv=cv, scoring="r2", n_jobs=-1)
        return scores

    def predict(self, model_name, input_df):
        """Predict on new input data."""
        model = self.models[model_name]
        return model.predict(input_df)

    def _score(self, name, y_pred):
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {
            "Model": name,
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3),
            "R²": round(r2, 3),
        }

    def save_models(self, path="models/"):
        import os
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            joblib.dump(model, f"{path}{safe_name}.pkl")
        print(f"Saved {len(self.models)} models to {path}")


if __name__ == "__main__":
    from data_generator import generate_crop_yield_data

    df = generate_crop_yield_data(2000)
    pipe = CropYieldPipeline(df)
    pipe.split_data()

    print("\n=== Supervised Models ===")
    print(pipe.train_supervised())

    print("\n=== Neural Network ===")
    print(pipe.train_neural_network())

    print("\n=== Semi-Supervised ===")
    print(pipe.train_semi_supervised(labeled_ratio=0.2))

    print("\n=== Unsupervised Clusters ===")
    print(pipe.get_unsupervised_clusters().head())