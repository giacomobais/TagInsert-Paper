from src.utils.utils import prepare_data, load_config, TaggingDataset

if __name__ == '__main__':
    config = load_config("config/VT_config.yaml")
    # train_file = "data/POS/raw/fake_translate_train.tagged"
    # val_file = "data/POS/raw/fake_translate_train.tagged"
    # test_file = "data/POS/raw/fake_translate_train.tagged"
    train_file = "data/POS/raw/new_sec_01-22.tagged"
    val_file = "data/POS/raw/new_sec_00.tagged"
    test_file = "data/POS/raw/new_sec_00.tagged"
    prepare_data(train_file, val_file, test_file, config['block_size'])

