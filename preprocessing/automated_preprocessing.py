import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE


def preprocessing_pipeline_heart(df: pd.DataFrame):
    print("📥 Memulai preprocessing untuk heart.csv...")

    # === 1. Definisikan fitur
    target_col = "HeartDisease"
    num_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
    cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

    # === 2. Split data (hindari data leakage)
    X = df.drop(columns=target_col)
    y = df[target_col]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === 3. Scaling numerikal
    scaler = StandardScaler()
    X_train_num = pd.DataFrame(
        scaler.fit_transform(X_train_raw[num_cols]),
        columns=num_cols,
        index=X_train_raw.index,
    )
    X_test_num = pd.DataFrame(
        scaler.transform(X_test_raw[num_cols]),
        columns=num_cols,
        index=X_test_raw.index,
    )

    # === 4. Encoding kategorikal
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train_raw[cat_cols])

    X_train_cat = pd.DataFrame(
        encoder.transform(X_train_raw[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train_raw.index,
    )
    X_test_cat = pd.DataFrame(
        encoder.transform(X_test_raw[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_test_raw.index,
    )

    # === 5. Gabungkan numerikal + kategorikal
    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

    # === 6. SMOTE (hanya pada train set)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)

    # === 7. Simpan ke direktori final submission

    output_dir = "preprocessing/models"
    os.makedirs(output_dir, exist_ok=True)
    # Simpan dataset
    pd.concat(
        [
            pd.DataFrame(X_train_resampled, columns=X_train_final.columns),
            y_train_resampled.reset_index(drop=True),
        ],
        axis=1,
    ).to_csv(os.path.join(output_dir, "train.csv"), index=False)

    pd.concat(
        [X_test_final.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    ).to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # Simpan scaler & encoder
    joblib.dump(scaler, os.path.join(output_dir, "standard_scaler.joblib"))
    joblib.dump(encoder, os.path.join(output_dir, "onehot_encoder.joblib"))

    joblib.dump(
        (X_train_resampled, X_test_final, y_train_resampled, y_test),
        os.path.join(output_dir, "data_preprocessed.joblib"),
    )
    print(f"✅ Semua artefak disimpan ke: {output_dir}")
    return


# === Eksekusi ===
if __name__ == "__main__":
    df = pd.read_csv("datasets/heart.csv")
    preprocessing_pipeline_heart(df)
