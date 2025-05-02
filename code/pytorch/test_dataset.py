from dataset_pytorch import create_partitioned_datasets

def main():
    # Get the partitioned datasets for all clients (farms)
    partitions = create_partitioned_datasets()
    print(f"Total number of clients: {len(partitions)}\n")
    # Print the shape of the features (X) and targets (y) for each client
    for cid, (X_client, y_client) in partitions.items():
        print(f"Client ID: {cid}")
        print(f"  Features shape (X): {X_client.shape}")
        print(f"  Targets shape (y): {y_client.shape}\n")

if __name__ == "__main__":
    main()
