import autorootcwd
import os
from src.utils.registry import DATASET_REGISTRY
from src.utils.visualize_dataloader import visualize_flow_dataset
from src.data.xca_flow import XCAFlow_DataModule  # noqa: F401

def main():
    # results 폴더 생성
    os.makedirs("results", exist_ok=True)
    
    # 데이터 모듈 생성 및 설정
    print("Creating XCA Flow DataModule...")
    dm = DATASET_REGISTRY.get("xca_flow")()
    dm.setup()
    
    # 각 데이터로더 확인
    print("\nChecking train dataloader...")
    visualize_flow_dataset(dm.train_dataloader(), "xca_flow_train", num_samples=10)
    
    print("\nChecking val dataloader...")
    visualize_flow_dataset(dm.val_dataloader(), "xca_flow_val")
    
    print("\nChecking test dataloader...")
    visualize_flow_dataset(dm.test_dataloader(), "xca_flow_test")
    
    print("\n✓ All checks completed! Results saved in results/ folder.")

if __name__ == "__main__":
    main()

