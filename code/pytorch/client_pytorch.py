import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import joblib
from sklearn.metrics import r2_score, mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import iqr
import copy
import pandas as pd
import os
from config import config
from dataset_pytorch import create_partitioned_datasets  
from model_pytorch_multioutput import create_model 


seed = config["seed"]
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
METRICS_TRAIN = []
METRICS_EVAL = []

# Configuration parameters
soil_properties_columns = config["soil_properties"]
model_params = config["model_params"]
batch_size = model_params["batch_size"]
local_epochs = model_params["epochs"]
early_stop_patience = model_params["early_stop_patience"]
val_ratio = config["val_ratio"]


def RMSE(y_true, y_pred):
    """Computes Root Mean Squared Error (RMSE)."""
    return np.sqrt(mse(y_true, y_pred))

def RPIQ(y_true, y_pred):
    """Computes Relative Percent Interquartile Range (RPIQ)."""
    return iqr(y_true, axis=0) / RMSE(y_true, y_pred)

def eval_learning(y_true, y_pred):
    """
    Evaluates the model by calculating R2, RMSE, and RPIQ for each soil property.
    Args:
        y_true (np.array): Actual soil property values.
        y_pred (np.array): Predicted soil property values.
    Returns:
        tuple: Arrays of R2, RMSE, and RPIQ values for each target column.
    """
    r2 = []
    rmse_vals = []
    rpiq_vals = []
    for col_idx in range(y_true.shape[1]):
        y_col_true = y_true[:, col_idx]
        y_col_pred = y_pred[:, col_idx]
        r2.append(r2_score(y_col_true, y_col_pred))
        rmse_vals.append(np.sqrt(mse(y_col_true, y_col_pred)))
        rpiq_vals.append(iqr(y_col_true) / np.sqrt(mse(y_col_true, y_col_pred)))
    return r2, rmse_vals, rpiq_vals


def get_model_weights(model):
    """Returns the model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_weights(model, weights):
    """Sets the model parameters from a list of NumPy arrays."""
    state_dict = model.state_dict()
    new_state_dict = {}
    for key, weight in zip(state_dict.keys(), weights):
        new_state_dict[key] = torch.tensor(weight)
    model.load_state_dict(new_state_dict)


def compute_weight_diff(weights_before, weights_after):
    """Compute the L2 norm difference between two lists of weights."""
    diff = 0.0
    for w1, w2 in zip(weights_before, weights_after):
        diff += np.sum((w1 - w2) ** 2)
    return np.sqrt(diff)

def train(model, optimizer, criterion, train_loader, val_loader, epochs, early_stop_patience, device):
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_epoch_loss:.4f}")

        # Evaluate on validation set
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{epochs} - Validation loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return best_val_loss


class FlowerClient(fl.client.NumPyClient):
    """
    Flower Client class for federated learning using PyTorch.
    """
    def __init__(self, model, X_train, y_train, X_val, y_val, cid, device):
        self.model = model.to(device)
        self.device = device
        # Convert data (pandas DataFrame or NumPy) to torch tensors
        self.X_train = torch.tensor(X_train.values, dtype=torch.float32) if hasattr(X_train, "values") else torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train.values, dtype=torch.float32) if hasattr(y_train, "values") else torch.tensor(y_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val.values, dtype=torch.float32) if hasattr(X_val, "values") else torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val.values, dtype=torch.float32) if hasattr(y_val, "values") else torch.tensor(y_val, dtype=torch.float32)
        scaler_path = config.get("scaler_path")
        self.best_global_val_loss = float("inf")
        self.best_global_state    = None
        if os.path.exists(scaler_path):
            self.scaler_y = joblib.load(scaler_path)
            print(f"Client {cid}: loaded y-scaler with mean={self.scaler_y.mean_} scale={self.scaler_y.scale_}")
        else:
            self.scaler_y = None
            print(f"Client {cid}: no scaler found at {scaler_path}, proceeding without scaling.")

        self.y_train_orig = y_train.values
        self.y_val_orig   = y_val.values

        # Scale once up front if scaler exists
        if self.scaler_y is not None:
            # transform both columns together
            y_train_scaled = self.scaler_y.transform(self.y_train_orig)
            y_val_scaled   = self.scaler_y.transform(self.y_val_orig)
            self.y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
            self.y_val   = torch.tensor(y_val_scaled,   dtype=torch.float32)
        else: 
            self.y_train = self.y_train_orig
            self.y_val= self.y_val_orig
        
        self.cid = cid
        self.local_round = 0
        self.criterion = nn.MSELoss()
        self.tmp = 1
        self.tmpa = 1
        # Initialize optimizer
        optimizer_name = model_params["optimizer"].lower()
        lr = model_params["learning_rate"]
        momentum = model_params.get("momentum", 0)
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError("Unsupported optimizer")

    def get_parameters(self, config):
        """Return local model weights to the server."""
        return get_model_weights(self.model)

    def fit(self, parameters, config):
        # Save initial weights for comparison
        initial_weights = get_model_weights(self.model)
        set_model_weights(self.model, parameters)
        print(f"Client {self.cid} - Starting local training")
        
        epochs = config.get("local_epochs", local_epochs)
        b_size = config.get("batch_size", batch_size)
        train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        val_dataset = torch.utils.data.TensorDataset(self.X_val, self.y_val)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=b_size, shuffle=False)
        
        train_loss = train(self.model, self.optimizer, self.criterion, train_loader, val_loader, epochs, early_stop_patience, self.device)
        print(f"Client {self.cid} - Training completed. Final validation loss: {train_loss:.4f}")
        if train_loss < self.best_global_val_loss:
            self.best_global_val_loss = train_loss
            self.best_global_state    = copy.deepcopy(self.model.state_dict())
            out_model = f"outputs/clients_models/client{self.cid}_best_overall.pth"
            os.makedirs(os.path.dirname(out_model), exist_ok=True)
            torch.save(self.best_global_state, out_model)
            print(f"Client {self.cid} → saved new overall best (loss {train_loss:.4f})")

        # Compare initial and final weights
        final_weights = get_model_weights(self.model)
        weight_diff = compute_weight_diff(initial_weights, final_weights)
        print(f"Client {self.cid} - Weight L2 norm difference after training: {weight_diff:.4f}")
        
        # Evaluate on validation set
        self.model.eval()


        predictions = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(self.device)
                predictions.append(self.model(X_batch).cpu().numpy())
        y_pred = np.vstack(predictions) 
        y_true = self.y_val.cpu().numpy()
        
        if self.scaler_y:

            print('Unscaling y_train/y_val data ...')
            y_pred = self.scaler_y.inverse_transform(y_pred)
            y_true = self.scaler_y.inverse_transform(y_true)

        r2_trns, rmse_trns, rpiq_trns = eval_learning(y_true, y_pred)
        metrics_dict = {}
        for i, prop in enumerate(soil_properties_columns):
            metrics_dict[f"R2-{prop}"] = round(r2_trns[i],2)
            metrics_dict[f"RMSE-{prop}"] = round(rmse_trns[i],2)
            metrics_dict[f"RPIQ-{prop}"] = round(rpiq_trns[i],2)
        metrics_dict["val_loss"] = round(train_loss, 4)
        metrics_dict["client_id"] = self.cid
        metrics_dict["round"] = self.local_round
        METRICS_TRAIN.append(metrics_dict)
        out_path = f'outputs/clients_metrics_train/client{self.cid}_train_metrics.csv'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(METRICS_TRAIN).to_csv(out_path, index=False)
        print(f"Client {self.cid} - Local training metrics: {metrics_dict}")
        self.local_round += 1
        
        return get_model_weights(self.model), len(self.X_train), metrics_dict

    def evaluate(self, parameters, config):
        """
        Evaluate the local model on the validation set.
        Args:
            parameters (list): Model weights for evaluation.
            config (dict): Configuration parameters from the server.
        Returns:
            tuple: Loss, number of validation examples, and evaluation metrics.
        """
        set_model_weights(self.model, parameters)
        print(f"Client {self.cid} - Local evaluation after server aggregation")

        b_size = config.get("batch_size", batch_size)
        val_dataset = torch.utils.data.TensorDataset(self.X_val, self.y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=b_size, shuffle=False)

        self.model.eval()
        losses = []
        predictions = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                losses.append(loss.item())
                predictions.append(outputs.cpu().numpy())
        avg_loss = np.mean(losses)
        y_pred = np.concatenate(predictions, axis=0)
        y_true = self.y_val.cpu().numpy()

        if self.scaler_y:
            # In case there is a scaler, the y_train/y_val data are already scaled. 
            # So, you need to unscale to calculate the metrics.
            print('Unscale y_train/y_val data in evaluate func ...')
            y_pred = self.scaler_y.inverse_transform(y_pred)
            y_true = self.scaler_y.inverse_transform(y_true)
        
        r2_vals, rmse_vals, rpiq_vals = eval_learning(y_true, y_pred)
        metrics_dict_eval = {}
        for i, prop in enumerate(soil_properties_columns):
            metrics_dict_eval[f"R2-{prop}"] = round(r2_vals[i],2)
            metrics_dict_eval[f"RMSE-{prop}"] = round(rmse_vals[i],2)
            metrics_dict_eval[f"RPIQ-{prop}"] = round(rpiq_vals[i],2)
        metrics_dict_eval["val_loss"]: round(avg_loss,4)
        metrics_dict_eval["client_id"] = self.cid
        metrics_dict_eval["round"] = self.local_round
        METRICS_EVAL.append(metrics_dict_eval)
        out_path = f'outputs/clients_metrics_eval/client{self.cid}_eval_nomodel.csv'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(METRICS_EVAL).to_csv(out_path, index=False)
        print(f"Client {self.cid} - Local eval metrics: {metrics_dict_eval}")

        return avg_loss, len(self.X_val), metrics_dict_eval

# --------------------------------------------------------------------------
# Main Function to Run Client
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", "-cid", type=int, default=0, help="Client ID (e.g., 0, 1, 2, ...)")
    args = parser.parse_args()
    cid = args.cid

    # Load data for the specified client
    X_client, y_client = create_partitioned_datasets(cid=cid)
    print(f"[Client {cid}] Received data with shape X={X_client.shape}, y={y_client.shape}")

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_client, y_client, test_size=val_ratio, random_state=config["seed"], shuffle=True
    )
    print(f"[Client {cid}] Train size: {X_train.shape}, Val size: {X_val.shape}")
    input_shape = (X_train.shape[1], 1) 
    output_shape = y_train.shape[1]      
    model = create_model(input_shape=input_shape, output_shape=output_shape)

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = FlowerClient(model, X_train, y_train, X_val, y_val, cid=cid, device=device)

    # Start the client and connect to the server
    fl.client.start_numpy_client(server_address=config["server_address"], client=client)

if __name__ == "__main__":
    main()
