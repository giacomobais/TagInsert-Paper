from src.utils.preprocess_utils import load_config, prepare_data, prepare_data_PMB

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
    pmb_file = "data/PMB/PMB_3.0.0/00/00/stagged/00.stagged"
    tagging = "CCG"
    # prepare_data(train_file, val_file, test_file, tagging, config)
    prepare_data_PMB(pmb_file)

