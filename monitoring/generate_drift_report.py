import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_preprocessed.parquet"
OUTPUT_DIR = SCRIPT_DIR


if not DATA_PATH.exists():
    print(f"Error: Data file not found at {DATA_PATH}")
    sys.exit(1)

df = pd.read_parquet(DATA_PATH)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Data loading complete. Reference: {len(X_train)} samples. Current: {len(X_test)} samples.")

report = Report(metrics=[
    DataDriftPreset(),
])

print("Executing data drift analysis...")
report.run(reference_data=X_train, current_data=X_test)

report_path = OUTPUT_DIR / "drift_report.html"
report.save_html(str(report_path))

print(f"Analysis completed. Technical report saved at: {report_path}")