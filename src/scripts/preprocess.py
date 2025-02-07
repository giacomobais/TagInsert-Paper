from src.utils.preprocess_utils import load_config, prepare_data

if __name__ == '__main__':
    config = load_config("config/VT_config.yaml")
    # train_file = "data/POS/raw/fake_translate_train.tagged"
    # val_file = "data/POS/raw/fake_translate_train.tagged"
    # test_file = "data/POS/raw/fake_translate_train.tagged"
    # train_file = "data/POS/raw/new_sec_01-22.tagged"
    # val_file = "data/POS/raw/new_sec_00.tagged"
    # test_file = "data/POS/raw/new_sec_00.tagged"
    train_file = "data/CCG/raw/Rebank_sec01-22.stagged"
    val_file = "data/CCG/raw/Rebank_sec00.stagged"
    test_file = "data/CCG/raw/Rebank_sec23.stagged"
    tagging = "CCG"
    prepare_data(train_file, val_file, test_file, tagging, config['block_size'], keep_proportion=float(config['data_proportion']))

