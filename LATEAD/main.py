import argparse
from data.data_preprocessing import preprocess_data
from models.slm_finetune import fine_tune_slm
from experiments.run_full_shot import run_full_shot
from experiments.run_few_shot import run_few_shot
import yaml

def main():
    parser = argparse.ArgumentParser(description="LSLAED Event Detection")
    parser.add_argument("--mode", choices=["preprocess", "train", "full_shot", "few_shot"], required=True)
    args = parser.parse_args()
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    if args.mode == "preprocess":
        preprocess_data(config["data"]["datasets"]["csed"], config["data"]["raw_dir"], config["data"]["processed_dir"])
    elif args.mode == "train":
        fine_tune_slm(config)
    elif args.mode == "full_shot":
        run_full_shot(config)
    elif args.mode == "few_shot":
        run_few_shot(config)

if __name__ == "__main__":
    main()