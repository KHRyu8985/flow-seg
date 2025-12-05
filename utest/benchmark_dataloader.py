import autorootcwd  # noqa: F401
import os
import time
import torch
from src.utils.registry import DATASET_REGISTRY
from src.data.xca_flow import XCAFlow_DataModule  # noqa: F401

def main():
    print("Creating XCA Flow DataModule...")
    dm = DATASET_REGISTRY.get("xca_flow")()
    dm.setup()
    
    train_loader = dm.train_dataloader()
    dataset_size = len(dm.train_dataset)
    batch_size = dm.train_bs
    
    print(f"\nDataset info:")
    print(f"  Total samples: {dataset_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches per epoch: {len(train_loader)}")
    print(f"  Num workers: 8")
    
    # Warmup
    print("\nWarming up...")
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        _ = batch
    
    # Actual timing
    print("\nTiming one epoch...")
    start_time = time.time()
    
    batch_count = 0
    sample_count = 0
    for batch in train_loader:
        batch_count += 1
        sample_count += batch['image'].shape[0]
        # Simulate minimal processing
        _ = batch['image'].shape
        _ = batch['label'].shape
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nResults:")
    print(f"  Elapsed time: {elapsed:.2f} seconds")
    print(f"  Batches processed: {batch_count}")
    print(f"  Samples processed: {sample_count}")
    print(f"  Time per batch: {elapsed/batch_count:.4f} seconds")
    print(f"  Time per sample: {elapsed/sample_count:.4f} seconds")
    print(f"  Samples per second: {sample_count/elapsed:.2f}")
    
    # Estimate for full training
    print(f"\nEstimated time for 100 epochs: {elapsed * 100 / 60:.2f} minutes")
    print(f"Estimated time for 500 epochs: {elapsed * 500 / 60:.2f} minutes")

if __name__ == "__main__":
    main()

