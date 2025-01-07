from src.utils.utils import prepare_data, load_config, TaggingDataset

if __name__ == '__main__':
    config = load_config("config/config.yaml")
    train_file = "data/POS/raw/fake_translate_train.tagged"
    val_file = "data/POS/raw/fake_translate_train.tagged"
    test_file = "data/POS/raw/fake_translate_train.tagged"
    prepare_data(train_file, val_file, test_file, config['block_size'])

    train_dataset = TaggingDataset("data/POS/processed/train_data.pth")
    val_dataset = TaggingDataset("data/POS/processed/val_data.pth")
    test_dataset = TaggingDataset("data/POS/processed/test_data.pth")

