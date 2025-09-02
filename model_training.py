import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV, KFold
import category_encoders as ce
from scipy.stats import loguniform, uniform
from assets_data_prep import prepare_data

warnings.filterwarnings("ignore")

# Load and clean training data
raw_train_df = pd.read_csv("train.csv")
train_clean = prepare_data(raw_train_df, "train")

neigh_medians = train_clean.groupby('neighborhood')['distance_from_center'].median()

fallback_median = neigh_medians.dropna().median()

distance_dict = {
    "neighborhood_medians": neigh_medians.to_dict(),
    "fallback": fallback_median
}

joblib.dump(distance_dict, "neigh_dist_medians.pkl")


# Feature reduction
X = train_clean.drop(columns=["price", "area_x_elevator", "elevator", "area_per_room"]).copy()
y = train_clean["price"].copy()

# Define feature groups
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
skewed_cols = ["area", "monthly_arnona", "building_tax",
               "distance_from_center", "arnona_per_m2"]
skewed_cols = [c for c in skewed_cols if c in num_cols]

cat_te = ["neighborhood"]
cat_ohe = ["property_type"]

# Preprocessing and modeling pipeline
preprocess = ColumnTransformer([
    ("num", StandardScaler(), [c for c in num_cols if c not in skewed_cols]),
    ("pow", PowerTransformer(), skewed_cols),
    ("te", ce.TargetEncoder(), cat_te),
    ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_ohe)
])

pipe = Pipeline([
    ("prep", preprocess),
    ("model", ElasticNet(max_iter=15000, random_state=42))
])

# Hyperparameter search space
param_dist = {
    "prep__te__smoothing": loguniform(0.05, 10),
    "prep__te__min_samples_leaf": [1, 10, 30],
    "model__alpha": loguniform(1e-3, 1e2),
    "model__l1_ratio": uniform(0, 1)
}

cv10 = KFold(n_splits=10, shuffle=True, random_state=42)

# Train model
rnd = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=200,
                         scoring="neg_root_mean_squared_error",
                         cv=cv10, n_jobs=-1, random_state=42, verbose=1)
rnd.fit(X, y)

# Save best model
joblib.dump(rnd.best_estimator_, "trained_model.pkl")
