# automate_ahmadrusdianto.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE


def load_data(path: str):
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    print("📥 Memulai preprocessing untuk heart.csv...")

    target_col = "HeartDisease"
    num_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
    cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

    X = df.drop(columns=target_col)
    y = df[target_col]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_num = pd.DataFrame(
        scaler.fit_transform(X_train_raw[num_cols]),
        columns=num_cols,
        index=X_train_raw.index,
    )
    X_test_num = pd.DataFrame(
        scaler.transform(X_test_raw[num_cols]), columns=num_cols, index=X_test_raw.index
    )

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

    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)

    return X_train_resampled, X_test_final, y_train_resampled, y_test, scaler, encoder


def save_artifacts(
    X_train, X_test, y_train, y_test, scaler, encoder, output_dir="preprocessing/models"
):
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.concat(
        [pd.DataFrame(X_train), y_train.reset_index(drop=True)], axis=1
    )
    test_df = pd.concat([pd.DataFrame(X_test), y_test.reset_index(drop=True)], axis=1)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    joblib.dump(scaler, os.path.join(output_dir, "standard_scaler.joblib"))
    joblib.dump(encoder, os.path.join(output_dir, "onehot_encoder.joblib"))
    joblib.dump(
        (X_train, X_test, y_train, y_test),
        os.path.join(output_dir, "data_preprocessed.joblib"),
    )
    
    print(f"✅ Semua artefak disimpan ke: {output_dir}")


if __name__ == "__main__":
    df = load_data("datasets/heart.csv")
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data(df)
    save_artifacts(X_train, X_test, y_train, y_test, scaler, encoder)
    print("✅ Preprocessing selesai!")
