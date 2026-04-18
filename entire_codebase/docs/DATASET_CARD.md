# Dataset Card: SparkyFitness Recommendation Data

## Data Sources

The training data is built from:

- Food.com recipe and interaction exports in `data/`.
- SparkyFitness production interactions captured through prediction and
  feedback logging.
- Optional Chameleon Swift object storage copies for versioned raw and training
  data.

## Dataset Construction

The batch pipeline builds train/validation/test CSVs by:

- Loading raw recipe and interaction files.
- Computing nutrition, allergen, cuisine, and recipe metadata features.
- Building six PCA-style user history features.
- Merging logged production impressions and feedback where available.
- Writing dataset manifests with split sizes and hashes.

## Data Quality Gates

Quality checks run at three points:

- Ingestion: schema, null, duplicate, row-count, and range checks.
- Training set compilation: split size, label balance, null checks, and temporal
  integrity.
- Production: inference feature drift and live data quality checks.

## Privacy

- The ML service receives numeric user and recipe surrogates rather than
  emails, names, or device identifiers.
- Raw user history is summarized through history components instead of sending
  full history records to serving.
- Inference feature retention cleanup runs in the drift monitor.

## Known Bias and Risk Areas

- Popular recipes can receive more feedback, which can reinforce exposure bias.
- Dietary groups with few users may have noisier metric estimates.
- Synthetic bootstrap interactions are useful for demos, but production
  monitoring should distinguish them from real user traffic.

## Lineage

Each compiled dataset should include:

- Source data paths or Swift object keys.
- Train/validation/test split sizes.
- SHA-256 hashes for generated CSV files.
- Code version or git commit.
- Creation timestamp.

The training pipeline logs the dataset metadata and artifacts to MLflow with
the model run.
