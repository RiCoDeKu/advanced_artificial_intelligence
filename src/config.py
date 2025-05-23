class DataLoaderConfig:
    BATCH_SIZE = 100
    RESTART_MODE = False
    VALID_SPLIT_RATIO = 0.01  # 全体の1%を検証用に
    NUM_WORKERS = 8
    
    # MNIST用
    DATA_DIR = '../data/MNIST/'
    TRAIN_IMAGES_FILE = 'MNIST_train_images.pt'
    VALID_IMAGES_FILE = 'MNIST_valid_images.pt'
    TEMP_DIR = './temp/MNIST/'
    
    # CelebA用
    # DATA_DIR = '../data/CelebA/'
    # TRAIN_IMAGES_FILE = 'tinyCelebA_train_images.pt'
    # VALID_IMAGES_FILE = 'tinyCelebA_valid_images.pt'
    # TEMP_DIR = './temp/CelebA/'