from utils.train_utils import resume_training
from src.utils.preprocess_utils import load_config
from src.utils.eval_utils import evaluate
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--model", type=str, required=True, help="Name of the model")
    args = parser.parse_args()
    model_to_eval = args.model
    config = load_config(f"config/{model_to_eval}_config.yaml")
    tagging = config['tagging']
    lang = config['language']
    run = config['run']
    if model_to_eval == "VT":
        model_package = resume_training(model_path = f"models/VanillaTransformer_{tagging}_{lang}_{str(config['data_proportion'])}_{run}", config = config, model_name = "VT", tagging=tagging)
        evaluate(model=model_package[0], config = config, tagging=tagging)
    elif model_to_eval == "TI":
        model_package = resume_training(model_path = f"models/TagInsert_{tagging}_{lang}_{str(config['data_proportion'])}_{run}", config = config, model_name = "TI", tagging=tagging)
        evaluate(model=model_package[0], config = config, tagging=tagging)
    elif model_to_eval == "TIL2R":
        model_package = resume_training(model_path = f"models/TagInsertL2R_{tagging}_{lang}_{str(config['data_proportion'])}_{run}", config = config, model_name = "TIL2R", tagging=tagging)
        evaluate(model=model_package[0], config = config, tagging=tagging)
    