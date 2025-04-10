from src.utils.train_utils import train, resume_training, preprocess_and_train_BERT_Encoder
from src.utils.preprocess_utils import load_config

if __name__ == '__main__':
    # get user input
    model_to_train = input("Enter the model you want to train: \n 1: BERT Encoder\n 2: Vanilla Transfomer\n 3: TagInsert\n 4: TagInsert L2R\n")
    if model_to_train == "1":
        config = load_config("config/BE_config.yaml")
        tagging = config["tagging"]
        model = preprocess_and_train_BERT_Encoder(config, tagging)
    if model_to_train == "2": 
        config = load_config("config/VT_config.yaml")
        tagging = config["tagging"]
        model_package = resume_training(model_path = f"models/VanillaTransformer_{tagging}_{str(config['data_proportion'])}", config = config, model_name = "VT", tagging=tagging)
        model = train(model_package, config, tagging, save = True)
    if model_to_train == "3":
        config = load_config("config/TI_config.yaml")
        tagging = config["tagging"]
        model_package = resume_training(model_path = f"models/TagInsert_{tagging}_{str(config['data_proportion'])}", config = config, model_name = "TI", tagging=tagging)
        model = train(model_package, config, tagging, save = False)
    if model_to_train == "4":
        config = load_config("config/TIL2R_config.yaml")
        tagging = config["tagging"]
        model_package = resume_training(model_path = f"models/TagInsertL2R_{tagging}_{str(config['data_proportion'])}", config = config, model_name = "TIL2R", tagging=tagging)
        model = train(model_package, config, tagging, save = True)


    
    