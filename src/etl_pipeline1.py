import pandas as pd
import numpy as np
import os
import csv
import time
import logging
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

# ==================================================
# CONFIGURATION
# ==================================================

CONFIG = {
    "output_folder": "output",
    "report_folder": "reports",
    "log_folder": "logs",
    "database_load": False,
    "database_url": "sqlite:///etl_database.db"
}

# ==================================================
# CREATE FOLDERS
# ==================================================

os.makedirs(CONFIG["output_folder"], exist_ok=True)
os.makedirs(CONFIG["report_folder"], exist_ok=True)
os.makedirs(CONFIG["log_folder"], exist_ok=True)

# ==================================================
# LOGGING
# ==================================================

logging.basicConfig(
    filename=os.path.join(
        CONFIG["log_folder"],
        "pipeline.log"
    ),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==================================================
# LOAD DATASET
# ==================================================

def load_dataset(file_path):

    logging.info("Loading dataset")

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            sample = f.read(5000)

        delimiter = csv.Sniffer().sniff(
            sample
        ).delimiter

        print(
            f"\nDetected Delimiter: {repr(delimiter)}"
        )

        df = pd.read_csv(
            file_path,
            sep=delimiter,
            on_bad_lines="skip"
        )

        logging.info(
            "CSV/TSV file loaded"
        )

        return df

    except Exception:

        try:

            df = pd.read_excel(
                file_path
            )

            logging.info(
                "Excel file loaded"
            )

            return df

        except Exception as e:

            logging.error(str(e))
            raise e

# ==================================================
# VALIDATE DATA
# ==================================================

def validate_data(df):

    report = []

    report.append(
        f"Rows: {df.shape[0]}"
    )

    report.append(
        f"Columns: {df.shape[1]}"
    )

    if df.shape[1] < 2:

        raise Exception(
            "Dataset contains insufficient columns"
        )

    duplicate_columns = (
        df.columns.duplicated().sum()
    )

    report.append(
        f"Duplicate Columns: {duplicate_columns}"
    )

    report.append(
        f"Duplicate Rows: {df.duplicated().sum()}"
    )

    report.append(
        "\nColumn Names:"
    )

    report.append(
        list(df.columns)
    )

    report.append(
        "\nData Types:"
    )

    report.append(
        df.dtypes
    )

    report.append(
        "\nMissing Values:"
    )

    report.append(
        df.isnull().sum()
    )

    return report

# ==================================================
# CLEAN DATA
# ==================================================

def clean_data(df):

    logging.info("Cleaning started")

    df.columns = df.columns.str.strip()

    # Convert numeric-looking strings to numeric
    for col in df.columns:

        try:
            converted = pd.to_numeric(
                df[col],
                errors="coerce"
            )
            if converted.notna().sum() > (
                0.8 * len(df)

            ):
                df[col] = converted

        except:
            pass

    duplicates_before = len(df)

    df.drop_duplicates(inplace=True)

    duplicates_removed = (
        duplicates_before - len(df)
    )

    print(
        f"\nDuplicate Rows Removed: "
        f"{duplicates_removed}"
    )

    for col in list(df.columns):

        missing_percent = (
            df[col].isnull().mean() * 100
        )

        if missing_percent > 50:

            print(
                f"Dropped column: {col}"
            )

            df.drop(
                columns=[col],
                inplace=True
            )

            continue

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):

            if df[col].isnull().sum() > 0:

                df[col] = df[col].fillna(
                    df[col].median()
                )

        else:

            if df[col].isnull().sum() > 0:

                mode_value = df[col].mode()

                if len(mode_value) > 0:

                    df[col] = df[col].fillna(
                        mode_value.iloc[0]
                    )

    print(
        "\nMissing Values Handled"
    )

    return df

# ==================================================
# HANDLE OUTLIERS
# ==================================================

def handle_outliers(df):

    logging.info(
        "Outlier handling started"
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = np.where(
            df[col] < lower,
            lower,
            df[col]
        )

        df[col] = np.where(
            df[col] > upper,
            upper,
            df[col]
        )

    print(
        "\nOutlier Handling Completed"
    )

    return df

# ==================================================
# TRANSFORM DATA
# ==================================================

def transform_data(df):

    logging.info(
        "Transformation started"
    )

    # Store original numeric columns
    original_numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    # Convert boolean values
    df.replace(
        {
            True: 1,
            False: 0,
            "Yes": 1,
            "No": 0,
            "YES": 1,
            "NO": 0,
            "yes": 1,
            "no": 0
        },
        inplace=True
    )

    # Detect only object/string date columns
    for col in list(df.columns):

        if not (
            pd.api.types.is_object_dtype(df[col])
            or str(df[col].dtype) == "string"
        ):
            continue

        try:

            converted = pd.to_datetime(
                df[col],
                errors="coerce",
                infer_datetime_format=True
            )

            if converted.notna().sum() > (
                0.8 * len(df)
            ):

                print(
                    f"Date column detected: {col}"
                )

                df[col + "_Year"] = (
                    converted.dt.year
                )

                df[col + "_Month"] = (
                    converted.dt.month
                )

                df[col + "_Day"] = (
                    converted.dt.day
                )

                df.drop(
                    columns=[col],
                    inplace=True
                )

        except:
            pass

    # Detect categorical columns
    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "string"
            ]
        ).columns
    )

    print(
        "\nCategorical Columns:"
    )

    print(
        list(categorical_columns)
    )

    encoder_mapping = []

    for col in categorical_columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

        for i, value in enumerate(
            encoder.classes_
        ):

            encoder_mapping.append(
                [
                    col,
                    value,
                    i
                ]
            )

    mapping_df = pd.DataFrame(
        encoder_mapping,
        columns=[
            "Column",
            "Original_Value",
            "Encoded_Value"
        ]
    )

    mapping_df.to_csv(
        os.path.join(
            CONFIG["report_folder"],
            "encoding_mapping.csv"
        ),
        index=False
    )

    print(
        "\nEncoding Completed"
    )

    # Detect ID columns
    id_columns = [

        col

        for col in df.columns

        if "id" in col.lower()
    ]

    scale_columns = [

        col

        for col in original_numeric_columns

        if col in df.columns
        and col not in id_columns
    ]

    valid_scale_columns = []

    for col in scale_columns:

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):

            valid_scale_columns.append(
                col
            )

    print(
        "\nNumeric Columns:"
    )

    print(
        valid_scale_columns
    )

    if len(valid_scale_columns) > 0:

        scaler = StandardScaler()

        df[valid_scale_columns] = (
            scaler.fit_transform(
                df[valid_scale_columns]
            )
        )

    print(
        "\nScaling Completed"
    )

    logging.info(
        "Transformation completed"
    )

    return df
# ==================================================
# SAVE REPORT
# ==================================================

def save_report(report, df):

    report_path = os.path.join(
        CONFIG["report_folder"],
        "data_quality_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        for item in report:

            f.write(
                str(item)
            )

            f.write("\n\n")

        f.write(
            "\n====================\n"
        )

        f.write(
            "DATA PROFILE\n"
        )

        f.write(
            "====================\n\n"
        )

        f.write(
            str(
                df.describe(
                    include="all"
                )
            )
        )

    print(
        "\nData Quality Report Generated"
    )

# ==================================================
# SAVE OUTPUT
# ==================================================

def save_output(df):

    output_file = os.path.join(
        CONFIG["output_folder"],
        "processed_data.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nProcessed Data Saved"
    )

    print(
        "Location:",
        os.path.abspath(output_file)
    )

# ==================================================
# DATABASE LOAD
# ==================================================

def load_to_database(df):

    if CONFIG["database_load"]:

        try:

            engine = create_engine(
                CONFIG["database_url"]
            )

            df.to_sql(
                "processed_data",
                engine,
                if_exists="replace",
                index=False
            )

            print(
                "\nData loaded to database"
            )

        except Exception as e:

            print(
                "\nDatabase Error:"
            )

            print(e)

# ==================================================
# MAIN PIPELINE
# ==================================================

def run_pipeline():

    start_time = time.time()

    file_path = input(
        "Enter dataset path: "
    ).strip()

    if not os.path.exists(
        file_path
    ):

        print(
            "\nDataset not found"
        )

        return

    print(
        "\nLoading dataset..."
    )

    df = load_dataset(
        file_path
    )

    print(
        "\nDataset Loaded Successfully"
    )

    print(
        "Shape:",
        df.shape
    )

    report = validate_data(
        df
    )

    df = clean_data(
        df
    )

    df = handle_outliers(
        df
    )

    df = transform_data(
        df
    )

    print(
        "\nProcessed Dataset Preview:"
    )

    print(
        df.head()
    )

    save_output(df)

    save_report(
        report,
        df
    )

    load_to_database(df)

    end_time = time.time()

    execution_time = round(
        end_time - start_time,
        2
    )

    print(
        "\nPipeline Completed Successfully"
    )

    print(
        "Execution Time:",
        execution_time,
        "seconds"
    )

# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    run_pipeline()