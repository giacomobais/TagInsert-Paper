from src.utils.preprocess_utils import load_config, prepare_data

if __name__ == '__main__':
    config = load_config("config/TI_config.yaml")
    tagging = config["tagging"]
    # train_file = "data/POS/raw/fake_translate_train.tagged"
    # val_file = "data/POS/raw/fake_translate_train.tagged"
    # test_file = "data/POS/raw/fake_translate_train.tagged"
    # train_file = "data/POS/raw/new_sec_01-22.tagged"
    # val_file = "data/POS/raw/new_sec_00.tagged"
    # test_file = "data/POS/raw/new_sec_00.tagged"
    # train_file = f"data/{tagging}/raw/Rebank_sec01-22.stagged"
    # val_file = f"data/{tagging}/raw/Rebank_sec00.stagged"
    # test_file = f"data/{tagging}/raw/Rebank_sec23.stagged"
    train_file = f"data/{tagging}/{config['language']}/raw/{config['language']}_train.stagged"
    val_file = f"data/{tagging}/{config['language']}/raw/{config['language']}_val.stagged"
    test_file = f"data/{tagging}/{config['language']}/raw/{config['language']}_test.stagged"
    prepare_data(train_file, val_file, test_file, tagging, config)

