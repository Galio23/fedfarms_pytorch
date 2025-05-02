"""
server_pytorch.py

Sets up a Flower server in PyTorch that:
1. Loads hyperparameters from config.py.
2. Builds a global PyTorch model (from model_pytorch.py) for evaluation on a global test set.
3. Uses a custom SaveModelStrategy (with FedAvg or weighted averaging) to aggregate weights and log metrics.
4. Starts the federated learning server.

Example usage: python server_pytorch.py
"""

import os
import timeit
import random
import numpy as np
import pandas as pd
import torch
import flwr as fl
import joblib
from typing import Dict, Optional, Tuple, List, Union
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import iqr
from config import config
from model_pytorch_multioutput import CNN1D, create_model
from custom_aggregators_pytorch import aggregate_WgtAvg, aggregate_FedAvg

# Set random seeds for reproducibility
seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

soil_properties_columns = config["soil_properties"]
metrics_dir = config["metrics_dir"]

# --------------------------------------------------------------------------
# Utility Metric Functions
# --------------------------------------------------------------------------
def RMSE(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def RPIQ(y_true, y_pred):
    """Compute Relative Percent Interquartile Range (RPIQ)."""
    return iqr(y_true, axis=0) / RMSE(y_true, y_pred)

def eval_learning(y_true, y_pred):
    """
    Evaluates the model by calculating R2, RMSE, and RPIQ for each soil property.
    
    Args:
        y_true (np.ndarray): Actual values (shape: [n_samples, n_targets]).
        y_pred (np.ndarray): Predicted values (shape: [n_samples, n_targets]).
        
    Returns:
        tuple: (r2, rmse, rpiq), each a np.ndarray with one entry per target column.
    """
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    rmse = RMSE(y_true, y_pred)
    rpiq = RPIQ(y_true, y_pred)
    return r2, rmse, rpiq

# --------------------------------------------------------------------------
# Global Test Data Loading and Prediction
# --------------------------------------------------------------------------
def load_global_test_data():
    """
    Loads the global test dataset from config["server_data"], selecting:
      - Feature columns based on the satellite type (landsat or sentinel).
      - Target columns from config["soil_properties"].
    Returns:
        X_test (DataFrame): Features for global test.
        y_test (DataFrame): Targets for global test.
    """
    df = pd.read_csv(config["server_data"])
    target_cols = config["soil_properties"]
    y_test = df[target_cols].copy()
    satellite_type = config["satellite_type"].lower()
    if satellite_type == "landsat":
        band_cols = config.get("landsat_bands", [])
    elif satellite_type == "sentinel":
        band_cols = config.get("sentinel_bands", [])
    else:
        raise ValueError("Invalid satellite_type in config. Must be 'landsat' or 'sentinel'.")
    X_test = df[band_cols].copy()
    print("Global test X shape:", X_test.shape)
    print("Global test y shape:", y_test.shape)
    return X_test, y_test

def get_model_weights(model: torch.nn.Module):
    """Returns model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_weights(model: torch.nn.Module, weights: NDArrays):
    """Sets model parameters from a list of NumPy arrays."""
    state_dict = model.state_dict()
    new_state_dict = {}
    for key, weight in zip(state_dict.keys(), weights):
        new_state_dict[key] = torch.tensor(weight)
    model.load_state_dict(new_state_dict)

def predict_model(model: torch.nn.Module, X: pd.DataFrame, device: torch.device):
    """
    Performs prediction on X using the PyTorch model.
    
    Args:
        model: PyTorch model.
        X: DataFrame or ndarray of shape [n_samples, n_features].
        device: 'cpu' or 'cuda'.
    
    Returns:
        np.ndarray of predictions.
    """
    model.eval()
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    # If the model expects an extra channel dimension, you can unsqueeze here.
    #X_tensor = X_tensor.unsqueeze(-1)  # Now shape: (n_samples, n_features, 1)
    with torch.no_grad():
        outputs = model(X_tensor)
    return outputs.cpu().numpy()

# --------------------------------------------------------------------------
# Metrics Aggregation Helpers
# --------------------------------------------------------------------------
def simple_mean_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Compute a simple unweighted mean of each metric across clients.
    
    Args:
        metrics (list): Each element is (num_examples, metrics_dict) for a client.
    
    Returns:
        dict: Aggregated metrics across all clients.
    """
    if not metrics:
        raise ValueError("The metrics list is empty.")
    simple_avg_metrics = {}
    for metric in ["RMSE", "R2", "RPIQ"]:
        for sp in soil_properties_columns:
            key = f"{metric}-{sp}"
            values = [m[1][key] for m in metrics if key in m[1]]
            simple_avg_metrics[key] = np.mean(values) if values else float("nan")
    return simple_avg_metrics

def wgt_average_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Computes a weighted average of each metric across clients, weighted by the number of examples.
    
    Args:
        metrics (list): Each element is a tuple (num_examples, metrics_dict) for a client.
    
    Returns:
        dict: Aggregated weighted metrics across all clients.
    """
    if not metrics:
        raise ValueError("The metrics list is empty.")
    weighted_avg = {}
    for metric in ["RMSE", "R2", "RPIQ"]:
        for sp in soil_properties_columns:
            key = f"{metric}-{sp}"
            total_examples = sum(n for n, m in metrics if key in m)
            if total_examples == 0:
                weighted_avg[key] = float("nan")
            else:
                weighted_sum = sum(n * m[key] for n, m in metrics if key in m)
                weighted_avg[key] = weighted_sum / total_examples
    return weighted_avg

# --------------------------------------------------------------------------
# Strategy: SaveModelStrategy (FedAvg)
# --------------------------------------------------------------------------
class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    A custom FedAvg strategy that saves aggregated model weights and logs metrics after each round.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:
        """
        Aggregates the model weights from participating clients, logs metrics, and saves them.
        """
        # Extract (weights, num_examples, client_id) for each client
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics['client_id'])
            for _, fit_res in results
        ]
        if not weights_results:
            print(f"[Round {server_round}] No client results received. Skipping aggregation.")
            # Return initial parameters instead of self.parameters (which doesn't exist)
            return self.initial_parameters, {}
        
        # Aggregate using the chosen strategy
        if config["strategy"].lower() == "wgtavg":
            aggregated_weights = aggregate_WgtAvg(weights_results)
        elif config["strategy"].lower() == "fedavg":
            aggregated_weights = aggregate_FedAvg(weights_results)
        else:
            raise ValueError(f"Unknown strategy: {config['strategy']}")
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Save aggregated weights to disk
        save_dir = config["save_weights_dir"]
        os.makedirs(save_dir, exist_ok=True)
        np.savez(f"{save_dir}/round-{server_round}-weights.npz", *aggregated_weights)

        # Aggregate metrics from clients (weighted or simple average)
        metrics_list_round = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
        if not metrics_list_round:
            round_metrics = {}
        else:
            if config["strategy"].lower() == "wgtavg":
                round_metrics = wgt_average_metrics(metrics_list_round)
            else:
                round_metrics = simple_mean_metrics(metrics_list_round)
        print(f"[Round {server_round}] Aggregated client metrics: {round_metrics}")

        return parameters_aggregated, round_metrics


# --------------------------------------------------------------------------
# Global Evaluation Function
# --------------------------------------------------------------------------
metrics_list = []
best_val_loss = float("inf")
best_round = -1

def compute_weight_norm(new_weights, initial_weights):
    total_norm = 0.0
    for new_w, init_w in zip(new_weights, initial_weights):
        total_norm += np.linalg.norm(new_w - init_w)
    return total_norm


def get_evaluate_fn(server_model: torch.nn.Module, device: torch.device):
    """
    Returns a function that Flower calls to evaluate the global model after each round.
    This version uses a precomputed y_scaler (loaded via joblib) to inverse-transform predictions.
    """
    X_test, y_test = load_global_test_data()

    # 1) Load one scaler per property into a dict:
    
    scaler_path = config.get('scaler_path')
    if scaler_path is not None:
        y_scaler = joblib.load(scaler_path)
        print(f"[Server] Loaded scaler for {soil_properties_columns} found!")
        print(f'Scaler mean: {y_scaler.mean_}')
    else:
        y_scaler= None
        print('Scaler not found for the Server!')

    initial_global_weights = get_model_weights(server_model)

    def evaluate(server_round: int, parameters, conf: Dict[str, Scalar]):
        global best_val_loss, best_round
        # Check whether parameters need conversion
        if hasattr(parameters, "tensors"):
            parameters_nd = parameters_to_ndarrays(parameters)
        else:
            parameters_nd = parameters
        set_model_weights(server_model, parameters_nd)
        
        # Optionally, log weight difference from initial global weights
        new_weights = get_model_weights(server_model)
        weight_diff = compute_weight_norm(new_weights, initial_global_weights)
        print(f"[Round {server_round}] Weight L2 norm difference from initial global model: {weight_diff:.4f}")


        # Predict on global test data and inverse-transform
        y_pred = predict_model(server_model, X_test, device)
        if y_scaler is not None:
            y_pred = y_scaler.inverse_transform(y_pred)
            y_true = y_test.values
            #y_true = y_scaler.inverse_transform(y_test.values)  # not needed for server
        else: 
            y_true = y_test.values
            

        total_loss = 0.0
        metrics_dict = {}
        for i, col in enumerate(soil_properties_columns):
            y_t = y_true[:, i].reshape(-1, 1)
            y_p = y_pred[:, i].reshape(-1, 1)
            loss = mean_squared_error(y_t, y_p)
            total_loss += loss
            metrics_dict[f"{col}_loss"] = float(loss)
            metrics_dict[f"{col}_R2"]   = float(r2_score(y_t, y_p))
            metrics_dict[f"{col}_RMSE"] = float(np.sqrt(loss))
            metrics_dict[f"{col}_RPIQ"] = float(iqr(y_t) / np.sqrt(loss) if loss>0 else 0)

        avg_loss = total_loss / len(soil_properties_columns)
        metrics_dict["round"] = server_round
        metrics_dict["average_loss"] = float(avg_loss)

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_round = server_round

            df_pred = pd.DataFrame(y_pred, columns=soil_properties_columns)
            df_true = y_test[soil_properties_columns].reset_index(drop=True)
            df_out = pd.concat([df_true.add_prefix("True_"), df_pred.add_prefix("Pred_")], axis=1)
            df_out["best_round"] = best_round
            os.makedirs(metrics_dir, exist_ok=True)
            df_out.to_csv(os.path.join(metrics_dir, "Pt_obs_pred_bestround.csv"), index=False)

        print(f"Best validation loss so far: round={best_round}, average_loss={best_val_loss:.4f}")
        metrics_list.append(metrics_dict)
        print(metrics_dict)
        return avg_loss, metrics_dict

    return evaluate


# --------------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------------
def main():
    print("[Server] Starting PyTorch FL server...")
    print('Current working directory:', os.getcwd())
    # Load global test data to determine input shape
    X_test, _ = load_global_test_data()
    num_features = X_test.shape[1]          
    input_shape = (num_features, 1)  
    print(f"[Server] Creating server model with input shape: {input_shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_model = create_model(input_shape, config["output_shape"]).to(device)
    
    # Load the pretrained centralized model if available
    pretrained_path = "code/models/best_model_notest.pth"
    if os.path.exists(pretrained_path):
        print(f"[Server] Loading pretrained model from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        server_model.load_state_dict(state_dict)
    else:
        print(f"[Server] Pretrained model not found at {pretrained_path}, using random initialization.")

    # Build the custom strategy with the initial parameters from the (pretrained) model

    strategy = SaveModelStrategy(
        min_available_clients=config["num_clients"],
        min_evaluate_clients=config["num_clients"],
        min_fit_clients=config["num_clients"],
        on_fit_config_fn=lambda rnd: {
            "batch_size": config["model_params"]["batch_size"],
            "local_epochs": config["model_params"]["epochs"],
            "momentum": config["model_params"].get("momentum", 0)
        },
        evaluate_fn=get_evaluate_fn(server_model, device),
        evaluate_metrics_aggregation_fn=simple_mean_metrics if config["strategy"].lower() == "fedavg" else wgt_average_metrics,
        initial_parameters=ndarrays_to_parameters(get_model_weights(server_model)),
    )

    num_rounds = config["num_rounds"]
    fl.server.start_server(
        server_address=config["server_address"],
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds)
    )

    metrics_df = pd.DataFrame(metrics_list)
    os.makedirs(config["metrics_dir"], exist_ok=True)
    metrics_df.to_csv(os.path.join(metrics_dir, "Total_server_evaluation_metrics.csv"), index=False)
    print("[Server] Training complete. Metrics saved.")


if __name__ == "__main__":
    execution_time = timeit.timeit(main, number=1)
    print(f"[Server] Execution time in seconds: {execution_time:.1f}")
