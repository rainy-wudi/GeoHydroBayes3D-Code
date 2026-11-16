# -*- coding: utf-8 -*-
"""
30B_Train_GroupMean_DNN_CorrectedRF_FinalGW.py

Purpose
-------
STEP 7: Retrain the corrected-RF GroupMean DNN.

This version DOES NOT try to match an unavailable/mismatched old split
using floating-point parameter vectors.

Instead it freezes a NEW corrected-RF 35/7/8 split keyed only by P_index,
using a deterministic seed (42). If the split CSV already exists, it is
reused verbatim on every later run.

Scientific target:
    f_DNN(theta) ~= E[h | theta]

Data:
    Group_Mean_Dataset_50_CorrectedRF_FinalGW.csv

Split:
    RandomState(42).permutation(P001...P050)
    first 35 -> Train
    next 7   -> Validation
    last 8   -> Test

Exact frozen IDs:
    Train:
        P014 P040 P031 P046 P018 P049 P027 P026 P033 P020
        P013 P005 P038 P009 P004 P007 P042 P047 P048 P016
        P010 P017 P025 P035 P032 P001 P045 P028 P034 P006
        P030 P012 P037 P002 P022

    Validation:
        P003 P044 P036 P024 P041 P011 P023

    Test:
        P019 P050 P021 P008 P043 P015 P029 P039

Architecture:
    6 -> 32 -> 64 -> 32 -> 23
    ReLU
    Dropout = 0.10

Training:
    Batch size = 8
    Adam lr = 1e-3
    weight decay = 1e-4
    MSE loss
    max epochs = 1500
    patience = 120
    seed = 42

Important
---------
- Scalers are fit ONLY on Train groups.
- Validation controls early stopping.
- Test groups are evaluated only after training finishes.
- No physics penalty is used.
"""

import os
import json
import random
import joblib
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================================
# 0. Reproducibility
# ============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================================
# 1. Paths
# ============================================================================

WORK_DIR = os.environ.get("GHB_WORK_DIR", os.path.dirname(os.path.abspath(__file__)))

GROUP_MEAN_PATH = os.path.join(
    WORK_DIR,
    "Group_Mean_Dataset_50_CorrectedRF_FinalGW.csv"
)

SIGMA_RF_PATH = os.path.join(
    WORK_DIR,
    "Sigma_RF_Pooled_23x23_CorrectedRF_FinalGW.csv"
)

SPLIT_PATH = os.path.join(
    WORK_DIR,
    "Group_Split_35_7_8_CorrectedRF.csv"
)

OUTPUT_DIR = os.path.join(
    WORK_DIR,
    "Step7_GroupMean_DNN_CorrectedRF_FinalGW"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================================
# 2. Columns
# ============================================================================

X_COLS = [
    "log10_K_Lower",
    "log10_K_Middle",
    "log10_K_Upper",
    "COV",
    "dh",
    "dv"
]

Y_COLS = [
    "zk101", "zk102", "zk103", "zk104", "zk105", "zk106",
    "zk107", "zk108", "zk109", "zk110", "zk120",
    "zk131", "zk132", "zk133", "zk134", "zk135", "zk136",
    "zk137", "zk138", "zk139", "zk140", "zk141", "zk410"
]


# ============================================================================
# 3. Hyperparameters
# ============================================================================

BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = int(os.environ.get("GHB_MAX_EPOCHS", "1500"))
PATIENCE = 120
DROPOUT = 0.10


# ============================================================================
# 4. Helpers
# ============================================================================

def safe_r2(y_true, y_pred):
    if np.var(y_true) <= 1e-15:
        return np.nan
    return r2_score(y_true, y_pred)


def safe_torch_load(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device
        )


# ============================================================================
# 5. Read corrected group-mean data
# ============================================================================

print("=" * 88)
print("STEP 7: CORRECTED-RF FINALGW GROUP-MEAN DNN")
print("=" * 88)

print("Group mean data :", GROUP_MEAN_PATH)
print("Sigma_RF        :", SIGMA_RF_PATH)
print("Frozen split    :", SPLIT_PATH)
print("Output          :", OUTPUT_DIR)

for path in [
    GROUP_MEAN_PATH,
    SIGMA_RF_PATH
]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


df = pd.read_csv(
    GROUP_MEAN_PATH
)

required_cols = [
    "P_index"
] + X_COLS + Y_COLS

missing = [
    c for c in required_cols
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        "Group mean data missing columns:\n"
        + ", ".join(missing)
    )

if len(df) != 50:
    raise RuntimeError(
        "Expected exactly 50 group rows, got %d."
        % len(df)
    )

df["P_index"] = df["P_index"].astype(int)

df = (
    df
    .sort_values("P_index")
    .reset_index(drop=True)
)

if list(df["P_index"]) != list(range(1, 51)):
    raise RuntimeError(
        "P_index must be exactly 1..50."
    )

if df[X_COLS + Y_COLS].isna().any().any():
    raise RuntimeError(
        "NaN detected in corrected group-mean data."
    )

if not np.isfinite(
    df[X_COLS + Y_COLS].to_numpy(float)
).all():
    raise RuntimeError(
        "Non-finite value detected in corrected group-mean data."
    )

print("\nCorrected group-mean structure: PASS (50 groups)")


# ============================================================================
# 6. Check corrected Sigma_RF
# ============================================================================

sigma_rf_df = pd.read_csv(
    SIGMA_RF_PATH,
    index_col=0
)

Sigma_RF = sigma_rf_df.loc[
    Y_COLS,
    Y_COLS
].to_numpy(float)

Sigma_RF = 0.5 * (
    Sigma_RF
    + Sigma_RF.T
)

eigvals = np.linalg.eigvalsh(
    Sigma_RF
)

if eigvals.min() <= 0:
    raise RuntimeError(
        "Sigma_RF is not positive definite. Min eigenvalue = %.6e"
        % eigvals.min()
    )

rf_std = np.sqrt(
    np.diag(Sigma_RF)
)

print("\nCorrected RF uncertainty:")
print("  Mean RF STD = %.4f m" % rf_std.mean())
print("  Min RF STD  = %.4f m" % rf_std.min())
print("  Max RF STD  = %.4f m" % rf_std.max())
print("  Min eig     = %.6e" % eigvals.min())


# ============================================================================
# 7. Create or reuse corrected-RF frozen split
# ============================================================================

if not os.path.exists(SPLIT_PATH):

    rng_split = np.random.RandomState(
        SEED
    )

    p_ids = df[
        "P_index"
    ].to_numpy(dtype=int)

    perm = rng_split.permutation(
        p_ids
    )

    train_ids = set(
        perm[:35].tolist()
    )

    val_ids = set(
        perm[35:42].tolist()
    )

    test_ids = set(
        perm[42:50].tolist()
    )

    split_df = df[
        ["P_index"] + X_COLS
    ].copy()

    def assign_split(p):
        p = int(p)

        if p in train_ids:
            return "Train"

        if p in val_ids:
            return "Validation"

        if p in test_ids:
            return "Test"

        raise RuntimeError(
            "Unexpected P_index: %d"
            % p
        )

    split_df["Split"] = split_df[
        "P_index"
    ].map(
        assign_split
    )

    split_df.to_csv(
        SPLIT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nNEW corrected-RF split created and frozen:")
    print("  %s" % SPLIT_PATH)

else:

    split_df = pd.read_csv(
        SPLIT_PATH
    )

    print("\nExisting corrected-RF frozen split reused:")
    print("  %s" % SPLIT_PATH)


# ============================================================================
# 8. Validate frozen split by P_index + parameter vector
# ============================================================================

required_split_cols = [
    "P_index"
] + X_COLS + [
    "Split"
]

missing_split = [
    c for c in required_split_cols
    if c not in split_df.columns
]

if missing_split:
    raise RuntimeError(
        "Frozen split missing columns:\n"
        + ", ".join(missing_split)
    )

if len(split_df) != 50:
    raise RuntimeError(
        "Frozen split must contain 50 rows, got %d."
        % len(split_df)
    )

split_df["P_index"] = split_df[
    "P_index"
].astype(int)

if set(split_df["P_index"]) != set(range(1, 51)):
    raise RuntimeError(
        "Frozen split P_index must be exactly 1..50."
    )

# Match by P_index; verify parameters have not changed.
check = df[
    ["P_index"] + X_COLS
].merge(
    split_df[
        ["P_index"] + X_COLS + ["Split"]
    ],
    on="P_index",
    suffixes=("_data", "_split"),
    how="left"
)

if check["Split"].isna().any():
    raise RuntimeError(
        "At least one P_index is missing in frozen split."
    )

for c in X_COLS:

    a = check[
        c + "_data"
    ].to_numpy(float)

    b = check[
        c + "_split"
    ].to_numpy(float)

    if not np.allclose(
        a,
        b,
        rtol=0.0,
        atol=1e-12
    ):
        raise RuntimeError(
            "Frozen split parameter mismatch for column: %s"
            % c
        )


split_map = dict(
    zip(
        split_df["P_index"].astype(int),
        split_df["Split"].astype(str)
    )
)

splits = np.array(
    [
        split_map[
            int(p)
        ]
        for p in df[
            "P_index"
        ]
    ],
    dtype=object
)

train_mask = (
    splits == "Train"
)

val_mask = (
    splits == "Validation"
)

test_mask = (
    splits == "Test"
)

counts = (
    int(train_mask.sum()),
    int(val_mask.sum()),
    int(test_mask.sum())
)

if counts != (
    35,
    7,
    8
):
    raise RuntimeError(
        "Frozen split must be 35/7/8, got %d/%d/%d."
        % counts
    )

print("\nFrozen split validation: PASS")
print("  Train      = 35")
print("  Validation = 7")
print("  Test       = 8")

print("\nTrain P_index:")
print(
    sorted(
        df.loc[
            train_mask,
            "P_index"
        ].astype(int).tolist()
    )
)

print("\nValidation P_index:")
print(
    sorted(
        df.loc[
            val_mask,
            "P_index"
        ].astype(int).tolist()
    )
)

print("\nTest P_index:")
print(
    sorted(
        df.loc[
            test_mask,
            "P_index"
        ].astype(int).tolist()
    )
)


# ============================================================================
# 9. Save group-mean data with split
# ============================================================================

with_split = df.copy()

with_split[
    "Split"
] = splits

WITH_SPLIT_PATH = os.path.join(
    OUTPUT_DIR,
    "Group_Mean_Dataset_50_CorrectedRF_WithSplit.csv"
)

with_split.to_csv(
    WITH_SPLIT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================================
# 10. Arrays
# ============================================================================

X = df[
    X_COLS
].to_numpy(float)

Y = df[
    Y_COLS
].to_numpy(float)

X_train = X[
    train_mask
]

Y_train = Y[
    train_mask
]

X_val = X[
    val_mask
]

Y_val = Y[
    val_mask
]

X_test = X[
    test_mask
]

Y_test = Y[
    test_mask
]


# ============================================================================
# 11. Train-only scaling
# ============================================================================

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_s = scaler_X.fit_transform(
    X_train
)

Y_train_s = scaler_Y.fit_transform(
    Y_train
)

X_val_s = scaler_X.transform(
    X_val
)

Y_val_s = scaler_Y.transform(
    Y_val
)

X_test_s = scaler_X.transform(
    X_test
)

SCALER_X_PATH = os.path.join(
    OUTPUT_DIR,
    "scaler_X_GroupMean_CorrectedRF.pkl"
)

SCALER_Y_PATH = os.path.join(
    OUTPUT_DIR,
    "scaler_Y_GroupMean_CorrectedRF.pkl"
)

joblib.dump(
    scaler_X,
    SCALER_X_PATH
)

joblib.dump(
    scaler_Y,
    SCALER_Y_PATH
)


# ============================================================================
# 12. Network
# ============================================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

print("\nDevice:", device)


class GroupMeanDNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 23)
        )

    def forward(self, x):
        return self.net(x)


Xtr_t = torch.tensor(
    X_train_s,
    dtype=torch.float32
)

Ytr_t = torch.tensor(
    Y_train_s,
    dtype=torch.float32
)

Xval_t = torch.tensor(
    X_val_s,
    dtype=torch.float32
)

Yval_t = torch.tensor(
    Y_val_s,
    dtype=torch.float32
)

Xtest_t = torch.tensor(
    X_test_s,
    dtype=torch.float32
)


generator = torch.Generator().manual_seed(
    SEED
)

loader = DataLoader(
    TensorDataset(
        Xtr_t,
        Ytr_t
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator
)


model = GroupMeanDNN().to(
    device
)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "GroupMean_DNN_CorrectedRF_Best.pth"
)


# ============================================================================
# 13. Train
# ============================================================================

train_losses = []
val_losses = []

best_val = np.inf
best_epoch = 0
no_improve = 0

print("\n" + "=" * 88)
print("TRAINING")
print("=" * 88)

for epoch in range(
    MAX_EPOCHS
):

    model.train()

    batch_losses = []

    for bx, by in loader:

        bx = bx.to(
            device
        )

        by = by.to(
            device
        )

        optimizer.zero_grad()

        pred = model(
            bx
        )

        loss = criterion(
            pred,
            by
        )

        loss.backward()

        optimizer.step()

        batch_losses.append(
            loss.item()
        )

    train_loss = float(
        np.mean(
            batch_losses
        )
    )

    model.eval()

    with torch.no_grad():

        val_pred = model(
            Xval_t.to(
                device
            )
        )

        val_loss = criterion(
            val_pred,
            Yval_t.to(
                device
            )
        ).item()

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    if val_loss < (
        best_val - 1e-8
    ):

        best_val = float(
            val_loss
        )

        best_epoch = epoch + 1

        no_improve = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

    else:

        no_improve += 1

    if (
        epoch == 0
        or
        (epoch + 1) % 50 == 0
    ):

        print(
            "Epoch [%4d/%4d] Train=%.6f Val=%.6f Best=%d"
            % (
                epoch + 1,
                MAX_EPOCHS,
                train_loss,
                val_loss,
                best_epoch
            )
        )

    if no_improve >= PATIENCE:

        print(
            "Early stopping after %d epochs without improvement."
            % PATIENCE
        )

        break


# ============================================================================
# 14. Restore best model and evaluate independent test
# ============================================================================

model.load_state_dict(
    safe_torch_load(
        BEST_MODEL_PATH,
        device
    )
)

model.eval()

with torch.no_grad():

    pred_test_s = model(
        Xtest_t.to(
            device
        )
    ).cpu().numpy()

pred_test = scaler_Y.inverse_transform(
    pred_test_s
)


# ============================================================================
# 15. Metrics
# ============================================================================

pooled_r2 = r2_score(
    Y_test.ravel(),
    pred_test.ravel()
)

uniform_r2 = r2_score(
    Y_test,
    pred_test,
    multioutput="uniform_average"
)

weighted_r2 = r2_score(
    Y_test,
    pred_test,
    multioutput="variance_weighted"
)

pooled_rmse = float(
    np.sqrt(
        mean_squared_error(
            Y_test.ravel(),
            pred_test.ravel()
        )
    )
)

pooled_mae = float(
    mean_absolute_error(
        Y_test.ravel(),
        pred_test.ravel()
    )
)


per_bh_rows = []

for j, bh in enumerate(
    Y_COLS
):

    yt = Y_test[
        :,
        j
    ]

    yp = pred_test[
        :,
        j
    ]

    per_bh_rows.append({
        "Borehole":
            bh,

        "R2":
            safe_r2(
                yt,
                yp
            ),

        "RMSE_m":
            float(
                np.sqrt(
                    mean_squared_error(
                        yt,
                        yp
                    )
                )
            ),

        "MAE_m":
            float(
                mean_absolute_error(
                    yt,
                    yp
                )
            )
    })


per_bh_df = pd.DataFrame(
    per_bh_rows
)

PER_BH_PATH = os.path.join(
    OUTPUT_DIR,
    "GroupMean_CorrectedRF_Test_PerBorehole_Metrics.csv"
)

per_bh_df.to_csv(
    PER_BH_PATH,
    index=False,
    encoding="utf-8-sig"
)

mean_bh_r2 = float(
    np.nanmean(
        per_bh_df[
            "R2"
        ].to_numpy(float)
    )
)

mean_bh_rmse = float(
    per_bh_df[
        "RMSE_m"
    ].mean()
)

mean_bh_mae = float(
    per_bh_df[
        "MAE_m"
    ].mean()
)


# ============================================================================
# 16. Save predictions
# ============================================================================

test_p = df.loc[
    test_mask,
    "P_index"
].astype(int).to_numpy()

prediction_rows = []

for i in range(
    len(
        X_test
    )
):

    rec = {
        "P_index":
            int(
                test_p[i]
            )
    }

    for j, c in enumerate(
        X_COLS
    ):
        rec[
            c
        ] = X_test[
            i,
            j
        ]

    for j, bh in enumerate(
        Y_COLS
    ):

        rec[
            bh + "_Mean_Abaqus"
        ] = Y_test[
            i,
            j
        ]

        rec[
            bh + "_DNN"
        ] = pred_test[
            i,
            j
        ]

        rec[
            bh + "_Error"
        ] = (
            pred_test[
                i,
                j
            ]
            -
            Y_test[
                i,
                j
            ]
        )

    prediction_rows.append(
        rec
    )


PRED_PATH = os.path.join(
    OUTPUT_DIR,
    "GroupMean_CorrectedRF_Independent_Test_Predictions.csv"
)

pd.DataFrame(
    prediction_rows
).to_csv(
    PRED_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================================
# 17. Save history and metrics
# ============================================================================

history = pd.DataFrame({
    "Epoch":
        np.arange(
            1,
            len(train_losses) + 1
        ),

    "Train_MSE_scaled":
        train_losses,

    "Validation_MSE_scaled":
        val_losses
})

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "GroupMean_CorrectedRF_Training_History.csv"
)

history.to_csv(
    HISTORY_PATH,
    index=False,
    encoding="utf-8-sig"
)


metrics = {
    "dataset":
        "Group_Mean_Dataset_50_CorrectedRF_FinalGW.csv",

    "scientific_target":
        "E[h | theta]",

    "split_file":
        SPLIT_PATH,

    "split_rule":
        "RandomState(42) permutation of P_index 1..50; first35 Train, next7 Validation, last8 Test",

    "train_P_index":
        sorted(
            df.loc[
                train_mask,
                "P_index"
            ].astype(int).tolist()
        ),

    "validation_P_index":
        sorted(
            df.loc[
                val_mask,
                "P_index"
            ].astype(int).tolist()
        ),

    "test_P_index":
        sorted(
            df.loc[
                test_mask,
                "P_index"
            ].astype(int).tolist()
        ),

    "architecture":
        [6, 32, 64, 32, 23],

    "dropout":
        DROPOUT,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LR,

    "weight_decay":
        WEIGHT_DECAY,

    "loss":
        "MSE only",

    "physics_penalty":
        False,

    "seed":
        SEED,

    "actual_epochs":
        len(
            train_losses
        ),

    "best_epoch":
        best_epoch,

    "best_validation_mse_scaled":
        best_val,

    "pooled_r2":
        float(
            pooled_r2
        ),

    "uniform_average_r2":
        float(
            uniform_r2
        ),

    "variance_weighted_r2":
        float(
            weighted_r2
        ),

    "pooled_rmse_m":
        pooled_rmse,

    "pooled_mae_m":
        pooled_mae,

    "mean_borehole_r2":
        mean_bh_r2,

    "mean_borehole_rmse_m":
        mean_bh_rmse,

    "mean_borehole_mae_m":
        mean_bh_mae,

    "mean_rf_std_m":
        float(
            rf_std.mean()
        ),

    "minimum_rf_std_m":
        float(
            rf_std.min()
        ),

    "maximum_rf_std_m":
        float(
            rf_std.max()
        )
}


METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "GroupMean_CorrectedRF_Test_Metrics.json"
)

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================================
# 19. Final console report
# ============================================================================

print("\n" + "=" * 88)
print("STEP 7 COMPLETE")
print("=" * 88)

print("Best epoch           : %d" % best_epoch)
print("Pooled R2            : %.4f" % pooled_r2)
print("Uniform-average R2   : %.4f" % uniform_r2)
print("Variance-weighted R2 : %.4f" % weighted_r2)
print("Pooled RMSE          : %.4f m" % pooled_rmse)
print("Pooled MAE           : %.4f m" % pooled_mae)
print("Mean Borehole R2     : %.4f" % mean_bh_r2)
print("Mean Borehole RMSE   : %.4f m" % mean_bh_rmse)
print("Mean Borehole MAE    : %.4f m" % mean_bh_mae)
print("Mean RF STD          : %.4f m" % rf_std.mean())

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nNEXT:")
print(
    "STEP 8 = strict 5-fold group-level OOF surrogate-error analysis."
)
