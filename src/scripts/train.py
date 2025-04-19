from src.utils.train_utils import train, resume_training, preprocess_and_train_BERT_Encoder
from src.utils.preprocess_utils import load_config
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--model", type=str, required=True, help="Name of the model")
    args = parser.parse_args()
    model_to_train = args.model
    config = load_config(f"config/{model_to_train}_config.yaml")
    tagging = config['tagging']
    lang = config['language']
    run = config['run']
    if model_to_train == "VT":
        model_package = resume_training(model_path = f"models/VanillaTransformer_{tagging}_{str(config['data_proportion'])}", config = config, model_name = "VT", tagging=tagging)
        model = train(model_package, config, tagging, save = True)
    elif model_to_train == "TI":
        epoch = config['epoch']
        print(f"models/TagInsert_{tagging}_{str(config['data_proportion'])}_{run}_{epoch}", flush = True)
        model_package = resume_training(model_path = f"models/TagInsert_{tagging}_{str(config['data_proportion'])}_{run}_{epoch}", config = config, model_name = "TI", tagging=tagging)
        model = train(model_package, config, tagging, save = True)
    elif model_to_train == "BE":
        model = preprocess_and_train_BERT_Encoder(config, tagging)
    elif model_to_train == "TIL2R":
        model_package = resume_training(model_path = f"models/TagInsertL2R_{tagging}_{lang}_{str(config['data_proportion'])}", config = config, model_name = "TIL2R", tagging=tagging)
        model = train(model_package, config, tagging, save = True)


    
    