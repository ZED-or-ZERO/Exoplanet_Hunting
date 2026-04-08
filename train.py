import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.preprocessing import detrend_light_curve
from src.model import ExoplanetCNN

import os
import joblib


# --- Random Forest Model ---
def train_baseline_model():
    """
    Loads features, combines them, and trains Random Forest 
    and saves the trained model to disk.
    """
    print("Loading feature datasets...")
    features_path = 'data/processed/clean_features.csv'
    fft_path = 'data/processed/fft_features.csv'
    
    features_df = pd.read_csv(features_path)
    fft_df = pd.read_csv(fft_path)
    
    print("Merging datasets (Data Fusion)...")
    df_merged = pd.merge(features_df, fft_df, on=['id', 'class'])
    
    X = df_merged.drop(columns=['id', 'class'])
    y = df_merged['class']
    
    print("Splitting data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Initializing Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=111, random_state=42, n_jobs=-1)
    
    print("Training model... (This may take a minute)")
    model.fit(X_train, y_train)
    
    print("\nEvaluating model on Test set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Check a models folder for it does exist
    os.makedirs('models', exist_ok=True)
    
    model_save_path = 'models/rf_baseline.pkl'
    print(f"Serializing and saving model to {model_save_path}...")
    joblib.dump(model, model_save_path)
    
    print("Training Pipeline Complete.")



# --- CNN 1D Model --- 


def train_deep_learning_model():
    print("1. Loading raw flux data...")
    df_flux = pd.read_csv("data/processed/clean_flux.csv")
    
    X = df_flux.drop(columns=['id', 'class']).values
    y = df_flux['class'].values

    print("2. Train/Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("3. Applying Detrending (Signal Processing)...")
    X_train = np.array([detrend_light_curve(row) for row in X_train])
    X_test = np.array([detrend_light_curve(row) for row in X_test])

    print("4. Scaling features (Z-Score Normalization)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Saving the scaler, you will need it in app.py for the interface!
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/cnn_scaler.pkl')

    print("5. Converting to PyTorch Tensors...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ExoplanetCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).view(-1, 1)

            optimizer.zero_grad()         
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()              
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{epochs} — loss: {running_loss/len(train_loader):.4f}")

    print("Training finished.")

    # Save Model Weights
    weights_path = 'models/cnn_weights.pth'
    torch.save(model.state_dict(), weights_path)
    print(f"8. Training complete! Model weights saved to {weights_path}")


if __name__ == "__main__":
    train_baseline_model()
    train_deep_learning_model()