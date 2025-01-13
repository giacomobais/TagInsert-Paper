from src.utils.utils import load_config, train, resume_training

if __name__ == '__main__':
    # get user input
    model_to_train = input("Enter the model you want to train: \n 1: Fine-Tuned DistilBERT\n 2: Vanilla Transfomer\n 3: TagInsert\n 4: TagInsert L2R\n")
    tagging = input("Enter the tagging task for the model:\n 1: POS\n 2: CCG\n")
    if tagging == "1":
        tagging = "POS"
    if tagging == "2":
        tagging = "CCG"
    if model_to_train == "1":
        pass
    if model_to_train == "2": 
        config = load_config("config/VT_config.yaml")
        model_package = resume_training(model_path = f"models/VanillaTransformer_{tagging}", config = config, model_name = "VT", tagging=tagging)
        model = train(model_package, config, tagging, save = True)
    if model_to_train == "3":
        config = load_config("config/TI_config.yaml")
        model_package = resume_training(model_path = f"models/TagInsert_{tagging}", config = config, model_name = "TI", tagging=tagging)
        model = train(model_package, config, tagging, save = True)
    if model_to_train == "4":
        pass


    
    