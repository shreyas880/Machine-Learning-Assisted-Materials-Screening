import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from preprocessing import preprocess
from utils import *


X_train, X_val, X_test, y_train, y_val, y_test, target_scaler, feature_scaler, metadata_test = preprocess()

# Rather than hardcoding, im assigning the number of features to a variable so that i can freely change the feautures at any time without any errors
INPUT_DIMENSIONS = X_train.shape[1]
os.makedirs("plots/model", exist_ok=True)

model = Sequential([
    Input((INPUT_DIMENSIONS, )),

    Dense(512, activation='relu'),
    Dropout(0.2),

    Dense(256, activation='relu'),
    Dropout(0.2),

    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(64, activation='relu'),
    
    Dense(6)
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='mse',
    metrics=['mae']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

history_df = pd.DataFrame(history.history)
history_df.to_csv(

    "plots/model/training_history.csv",

    index=False

)


# Making the predictions and then saving the model
y_pred = model.predict(X_test)

model.save("best_material_model.keras")

# Converting the values back to original scale to make sense of it using ds
# This is also the reason ive passed the target_scaler in the preprocess function
y_pred = target_scaler.inverse_transform(y_pred)

y_test = target_scaler.inverse_transform(y_test)


# Create dataframes - just easier to work with

predictions = pd.DataFrame(y_pred, columns=TARGET_COLUMNS)

actual_values = pd.DataFrame(y_test, columns=TARGET_COLUMNS)


# Restting index to keep track of the materials so that we can screen and rank them later
metadata_test = metadata_test.reset_index(drop=True)
predictions = predictions.reset_index(drop=True)
actual_values = actual_values.reset_index(drop=True)

# Saving csvs
actual_values.to_csv("actual_values.csv", index=False)
predictions.to_csv("predicted_values.csv", index=False)

# Final Results 
results = pd.concat([
        metadata_test,
        predictions
    ], axis=1)

results.to_csv("material_predictions.csv",index=False)

print(results.head())

# Evaluation of the results starts here
R2_VALUES = []

RMSE_VALUES = []

MAE_VALUES = []

for i,column in enumerate(TARGET_COLUMNS):
    r2 = r2_score(y_test[:,i], y_pred[:,i])
    mae = mean_absolute_error(y_test[:,i], y_pred[:,i])
    rmse = root_mean_squared_error(y_test[:,i], y_pred[:,i])

    R2_VALUES.append(r2)
    RMSE_VALUES.append(rmse)
    MAE_VALUES.append(mae)

    print(column)
    print("R² =",r2)
    print("RMSE =",rmse)
    print("MAE =",mae)

metrics_df = pd.DataFrame({
    "Property":TARGET_COLUMNS,
    "R2":R2_VALUES,
    "RMSE":RMSE_VALUES,
    "MAE":MAE_VALUES
})

metrics_df.to_csv("plots/model/model_metrics.csv", index=False)

# Plotting and saving the various visual metrics to aid the evaluation of the model

def plot_loss_curve():
    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/model/loss_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def plot_metrics():
    metrics = {
        "R2_scores.png":R2_VALUES,
        "RMSE_scores.png":RMSE_VALUES,
        "MAE_scores.png":MAE_VALUES
    }


    for filename,values in metrics.items():
        plt.figure(figsize=(8,5))

        plt.bar(
            TARGET_COLUMNS,
            values
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.title(
            filename.replace(
                ".png",
                ""
            )
        )

        plt.tight_layout()

        plt.savefig(
            f"plots/model/{filename}",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

def plot_predictions():
    fig,axes = plt.subplots(2, 3, figsize=(14,8))
    axes = axes.flatten()

    for index,column in enumerate(TARGET_COLUMNS):
        axes[index].scatter(
            y_test[:,index],
            y_pred[:,index],
            alpha=0.6
        )

        axes[index].set_title(column)

        axes[index].set_xlabel("Actual")

        axes[index].set_ylabel("Predicted")

    plt.tight_layout()

    plt.savefig(
        "plots/model/predicted_vs_actual.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def generate_model_plots():
    plot_loss_curve()
    plot_metrics()
    plot_predictions()

#=========================================================
# MAIN
#=========================================================

if __name__ == "__main__":

    generate_model_plots()