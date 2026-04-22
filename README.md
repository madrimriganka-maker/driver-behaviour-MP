# Driver Behavior Anomaly Detection

This project classifies driving sessions into three behavior classes using smartphone/IMU-style motion sensor data:

- `NORMAL` - smooth, safe driving
- `AGGRESSIVE` - harsh braking, sharp cornering, rapid acceleration
- `SLOW` - drowsy or overly cautious driving

The workflow is implemented as a step-by-step notebook pipeline from data setup to model evaluation and unlabeled prediction.

## Project Pipeline

```text
00_Setup_Config
  -> 01_Data_Exploration
  -> 02_Preprocessing_Feature_Engineering
  -> 03_Windowing
  -> 04_Balance_Normalize
  -> 05_Model_Training
  -> 06_Evaluation_Prediction
```

## Method Summary

- Input signals: accelerometer + gyroscope channels (`AccX`, `AccY`, `AccZ`, `GyroX`, `GyroY`, `GyroZ`)
- Engineered features: jerk components, lateral q-force, and speed variance
- Total channels per timestep: 11
- Windowing: 5-second windows (`50` samples at `10 Hz`) with `50%` overlap
- Imbalance handling: SMOTE on training split
- Scaling: `StandardScaler`
- Model: CNN-LSTM hybrid (`Conv1D x2 -> LSTM x2 -> Dense -> Softmax`)
- Classes: 3 (`NORMAL`, `AGGRESSIVE`, `SLOW`)

## Repository Structure

- `00_Setup_Config.ipynb` - environment, paths, config, constants
- `01_Data_Exploration.ipynb` - EDA and class/signal visualization
- `02_Preprocessing_Feature_Engineering.ipynb` - cleaning and derived features
- `03_Windowing.ipynb` - sequence construction from continuous sensor stream
- `04_Balance_Normalize.ipynb` - train/val/test split, SMOTE, scaling
- `05_Model_Training.ipynb` - CNN-LSTM training and history export
- `06_Evaluation_Prediction.ipynb` - test evaluation, plots, unlabeled inference
- `test_motion_data.csv` - labeled dataset
- `test_motion_data_nolabels.csv` - unlabeled dataset for inference
- `outputs/` - saved model, arrays, plots, report files, predictions

## Setup

Use Python 3.10+ in a virtual environment (or conda env), then install:

```bash
pip install tensorflow scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
```

## How To Run

Run notebooks in numeric order (`00` to `06`), top-to-bottom in each notebook.

Artifacts are saved automatically in `outputs/`, including:

- `best_model.keras`
- `training_history.json`
- `classification_report.txt`
- `predictions_unlabeled.csv`
- intermediate arrays (`X_train.npy`, `y_train.npy`, etc.)
- diagnostic figures (confusion matrix, ROC curves, training curves, class plots)

## Results (Current Run)

From `outputs/classification_report.txt`:

- Test Accuracy: **68.42%**
- Test Loss: **0.7411**

Per-class metrics:

- `NORMAL`: precision `1.0000`, recall `0.1667`, F1 `0.2857`
- `AGGRESSIVE`: precision `0.6250`, recall `1.0000`, F1 `0.7692`
- `SLOW`: precision `0.7000`, recall `0.8750`, F1 `0.7778`

## Model Details

- Model name: `CNN_LSTM_DriverBehavior`
- Parameters: ~`92,035`
- Training epochs: up to `60` (with callbacks like early stopping/checkpointing)
- Class weighting used to reduce under-learning of minority behavior classes

## Notes

- `.ipynb_checkpoints/` contains Jupyter checkpoint copies and is not part of the core pipeline.
- You can swap in a larger dataset (same schema) and rerun notebooks for improved generalization.
