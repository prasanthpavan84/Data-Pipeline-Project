
The validation layer records:

- Number of rows and columns
- Duplicate columns
- Duplicate rows
- Column names
- Data types
- Missing values per column

### 3. Cleaning

The cleaning stage:

- Strips whitespace from column names
- Converts numeric-looking text columns into numeric columns
- Removes duplicate rows
- Drops columns where missing values exceed 50%
- Fills numeric nulls with the median
- Fills categorical nulls with the mode

### 4. Transformation

The transformation stage:

- Converts `Yes`/`No` and boolean values to `1`/`0`
- Detects date-like string columns and expands them into year, month, and day columns
- Encodes categorical columns with `LabelEncoder`
- Saves category encoding mappings to `reports/encoding_mapping.csv`
- Standard-scales numeric columns while excluding ID-like columns

### 5. Output

The pipeline writes:

- Processed dataset: `output/processed_data.csv`
- Data quality report: `reports/data_quality_report.txt`
- Encoding map: `reports/encoding_mapping.csv`
- Execution log: `logs/pipeline.log`

## Setup

### Prerequisites

- Python 3.10 or newer
- pip

### Installation

```bash
git clone https://github.com/<your-username>/Data-Pipeline-Project.git
cd Data-Pipeline-Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For macOS or Linux:

```bash
source .venv/bin/activate
```

## Usage

Run the ETL pipeline:

```bash
python src/etl_pipeline1.py
```

When prompted, enter the dataset path:

```bash
data/custumer.csv
```

After completion, review the processed files in `output/` and `reports/`.

## Optional Database Load

Database loading is controlled by the `CONFIG` dictionary in `src/etl_pipeline1.py`:

```python
CONFIG = {
    "output_folder": "output",
    "report_folder": "reports",
    "log_folder": "logs",
    "database_load": False,
    "database_url": "sqlite:///etl_database.db"
}
```

To enable database loading, set:

```python
"database_load": True
```

The default database URL writes to a local SQLite database named `etl_database.db`.

## Example Use Cases

- Customer churn preprocessing
- Business analytics data preparation
- Data quality reporting
- Machine learning feature preparation
- CSV/Excel dataset standardization before modeling

## Recommended GitHub Publishing Notes

Before publishing the project, consider:

- Keep small sample datasets only if they are safe to share.
- Avoid committing large generated outputs unless they are needed for demonstration.
- Keep secrets, database credentials, private data, and large raw data files out of Git.
- Rename `custumer.csv` to `customer.csv` if you want cleaner public naming.
- Consider adding tests if this pipeline will be reused in production workflows.

## Current Limitations

- The script is interactive and expects the dataset path through `input()`.
- Label encoding is saved, but the fitted encoders and scaler are not persisted as reusable model artifacts.
- Outlier handling is applied to all numeric columns after validation and cleaning.
- The pipeline currently writes fixed output filenames.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- SQLAlchemy
- openpyxl

## License

Add a license file before publishing if this repository will be shared publicly.