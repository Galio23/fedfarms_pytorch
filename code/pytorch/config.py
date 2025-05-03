"""
Configuration parameters for the federated learning experiments.
Adjust these values as needed.
"""
config = {
    # =======================================================
    # Data Configuration
    # =======================================================
    "criteria": "farm",  # Using 'farm'-based partitioning
    "satellite_type": "sentinel",  # Options: "landsat", "sentinel"
    "soil_properties": [
        "Clay_gkg_filtered",
        "C_gkg_filtered",
        
    ],
    # Define the list of bands for each satellite type
    "landsat_bands": ["blue", "green", "red", "nir", "swir1", "swir2"],
    "sentinel_bands": ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"],
    
    # Client Data path
    "client_data": "data/farms_sentinel.csv",
    "scaler_path": "code/models/scaler_y_2cols.pkl",
    # Server Data paths (if needed)
    "server_data": "data/test_global.csv",

    # =======================================================
    # Federated Learning / Flower Server Configuration
    # =======================================================
    "strategy" : "FedAvg",        # Options: "FedAvg", "WgtAvg"
    "server_address": "15.204.230.95:8080", #"15.204.230.95:8080",
    "num_rounds": 200,           # Total federated training rounds

    # =======================================================
    # Model Configuration
    # =======================================================
    "model_params": {
        "filters1": 64,
        "filters2": 64,
        "kernel_size1": 5,
        "kernel_size2": 5,
        "optimizer": "adam",      # Options: "adam", "sgd"
        "learning_rate": 0.001,
        "momentum": 0.1,          # For SGD with momentum
        "batch_size": 8,
        "epochs": 10,            # Local epochs per round
        "early_stop_patience": 10 # Patience for early stopping
    },

    "seed": 42, 
    "num_clients" : 50,                 
    "input_shape":  10,            
    "output_shape": 2,          
    "val_ratio": 0.3,            

    # Directories for saving model weights, metrics, etc.
    "save_weights_dir": "outputs/weights_server",
    "metrics_dir": "outputs/metrics_server"
}
