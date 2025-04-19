from src.utils.preprocess_utils import load_config, prepare_data

if __name__ == '__main__':
    config = load_config("config/TI_config.yaml")
    tagging = config["tagging"]
    train_file = f"data/{tagging}/{config['language']}/raw/{config['language']}_train.stagged"
    val_file = f"data/{tagging}/{config['language']}/raw/{config['language']}_val.stagged"
    test_file = f"data/{tagging}/{config['language']}/raw/{config['language']}_test.stagged"
    prepare_data(train_file, val_file, test_file, tagging, config)

