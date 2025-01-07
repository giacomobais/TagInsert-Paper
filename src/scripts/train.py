from src.utils.utils import load_config, train

if __name__ == '__main__':
    config = load_config("config/config.yaml")

    _ = train(config)


    
    