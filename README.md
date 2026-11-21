# CREA: Efficient Network Traffic Anomaly Detection under Evolving Normality

CREA addresses a practical problem in anomaly detection: a detector trained on historical normal behavior may assign high anomaly scores to legitimate behavior that emerges later. CREA maintains a reference of normal behavior, directs additional computation to selected observations, and uses a limited label budget to determine which samples can support a model update.

## Framework

![CREA framework](media/framework%20.png)

| Component | Function | Source |
| --- | --- | --- |
| Intrinsic Confidence Anchoring (ICA) | Uses a variational autoencoder and normal anchors to estimate geometric deviation, latent uncertainty, and reconstruction error. These signals yield a base risk score for routing and a confidence estimate for adaptation. | [`CREA_ICA.py`](admodels/CREA_ICA.py) |
| Dynamic Precision Routing (T-MDSS) | Applies a lightweight temporal convolutional network to each observation. Sample risk and window-level distribution shift determine which observations receive further analysis by a local Transformer. | [`CREA_Routing.py`](admodels/CREA_Routing.py) |
| Shift Explanation | Compares reference and current-window score distributions using a permutation test. When the test detects a difference, two soft masks identify historical samples to retain and current samples to prioritize for label queries. | [`CREA_Explanation.py`](admodels/CREA_Explanation.py) |
| Selective Adaptation | Uses retained historical samples to estimate parameter importance. Newly confirmed normal samples guide the update, while an importance-weighted penalty limits changes to parameters that support the historical reference. | [`CREA_Adaptation.py`](admodels/CREA_Adaptation.py) |

## Implementation

[`run_crea.py`](run_crea.py) connects the four components for a single adaptation window. It trains on initial normal records, calibrates the detector on a separate normal control set, and selects label queries from the current validation window. Only queried records confirmed as normal can enter the updated reference. The held-out test window is used for final scoring.

This repository provides a reimplementation of the single-window CREA workflow.

## Input data

The entry point accepts a preprocessed NumPy `.npz` file. Feature matrices use the same columns and preserve chronological order within each split.

| Array | Contents |
| --- | --- |
| `train` | Normal records for initial training |
| `control` | Separate historical normal records for reference calibration |
| `validation` | Current-window records available for scoring and label queries |
| `validation_labels` | One label per validation record: `0` for normal and `1` for anomalous |
| `test` | Held-out records for final scoring |

The implementation uses Python, NumPy, PyTorch, and scikit-learn. From the repository root, the following command lists the input and configuration options:

```bash
python run_crea.py --help
```

The output records the window-level drift statistic, permutation-test p-value, queried row indices, number of confirmed normal samples, update status, and held-out anomaly scores.

## Experiments and manuscript materials

The manuscript evaluates CREA on four datasets: Kyoto 2006+ and CIC IoT-DIAD 2024 for network intrusion detection, BGL for system log anomaly detection, and LANL-CMSCSE for authentication anomaly detection. The evaluation covers single and repeated adaptation across chronological windows and examines detection performance, label cost, and computational overhead.

