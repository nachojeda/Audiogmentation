import random
from preprocessing.utils import(
    create_dataset_splits,
    GTZANDataset,
    get_dataloader
)

if __name__ == "__main__":
    # Set random seed for reproducibility
    # random.seed(42)
    # dataset_path = "../../datasets/genres_original"
    # Split dataset into train, test and val folders
    # create_dataset_splits(dataset_path)

    # Load train data
    train_loader = get_dataloader(data_path='datasets/', split='train')
    iter_train_loader = iter(train_loader)
    train_wav, train_genre = next(iter_train_loader)

    # Load validation data
    valid_loader = get_dataloader(split='valid')

    # Load test data
    test_loader = get_dataloader(split='test')

    iter_test_loader = iter(test_loader)
    test_wav, test_genre = next(iter_test_loader)
    print('training data shape: %s' % str(train_wav.shape))
    print('validation/test data shape: %s' % str(test_wav.shape))
    print(train_genre)
