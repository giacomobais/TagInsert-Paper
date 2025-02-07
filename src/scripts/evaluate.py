from utils.train_utils import resume_training
from src.utils.preprocess_utils import load_config
from src.utils.eval_utils import evaluate

if __name__ == '__main__':
    model_to_eval = input("Enter the model you want to evaluate: \n 1: Fine-Tuned DistilBERT\n 2: Vanilla Transfomer\n 3: TagInsert\n 4: TagInsert L2R\n")
    tagging = input("Enter the tagging task for the model:\n 1: POS\n 2: CCG\n")
    if tagging == "1":
        tagging = "POS"
    elif tagging == "2":
        tagging = "CCG"

    if model_to_eval == "1":
        pass
    if model_to_eval == "2":
        config = load_config("config/VT_config.yaml")
        model_package = resume_training(model_path = f"models/VanillaTransformer_{tagging}_{str(config['data_proportion'])}", config = config, model_name = "VT", tagging=tagging)
        evaluate(model=model_package[0], config = config, tagging=tagging)
    if model_to_eval == "3":
        config = load_config("config/TI_config.yaml")
        model_package = resume_training(model_path = f"models/TagInsert_{tagging}_{str(config['data_proportion'])}", config = config, model_name = "TI", tagging=tagging)
        evaluate(model=model_package[0], config = config, tagging=tagging)
    if model_to_eval == "4":
        config = load_config("config/TIL2R_config.yaml")
        model_package = resume_training(model_path = f"models/TagInsertL2R_{tagging}_{str(config['data_proportion'])}", config = config, model_name = "TIL2R", tagging=tagging)
        evaluate(model=model_package[0], config = config, tagging=tagging)
    