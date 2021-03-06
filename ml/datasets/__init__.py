import tensorflow as tf
import tensorflow_datasets as tfds
from ml import get_base_dir


def get_dataset(dataset_name, as_supervised=True):
    base_path = get_base_dir()
    dataset_dir = base_path / 'datasets/storage'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data, info = tfds.load(name=dataset_name, data_dir=dataset_dir, with_info=True, shuffle_files=True,
                           as_supervised=as_supervised)
    if dataset_name == 'mnist':
        data_train, data_test = data['train'], data['test']
        assert isinstance(data_train, tf.data.Dataset)
        assert isinstance(data_test, tf.data.Dataset)
        assert info.features['image'].shape == (28, 28, 1)
        assert info.features['label'].num_classes == 10
        assert info.splits['train'].num_examples == 60000
        assert info.splits['test'].num_examples == 10000
    elif dataset_name == 'cifar10':
        data_train, data_test = data['train'], data['test']
        assert isinstance(data_train, tf.data.Dataset)
        assert isinstance(data_test, tf.data.Dataset)
        assert info.features['image'].shape == (32, 32, 3)
        assert info.features['label'].num_classes == 10
        assert info.splits['train'].num_examples == 50000
        assert info.splits['test'].num_examples == 10000
    elif dataset_name == 'cifar100':
        data_train, data_test = data['train'], data['test']
        assert isinstance(data_train, tf.data.Dataset)
        assert isinstance(data_test, tf.data.Dataset)
        assert info.features['image'].shape == (32, 32, 3)
        assert info.features['coarse_label'].num_classes == 20
        assert info.features['label'].num_classes == 100
        assert info.splits['train'].num_examples == 50000
        assert info.splits['test'].num_examples == 10000
    else:
        raise NameError(f'cannot find dataset {dataset_name}')
    return data, info


dataset_urls = {
    'fruits': 'fruits-360_dataset/fruits-360/',
    'shuihu': '水浒传.txt',
    'tencent_embedding_1m': 'word_vectors_1m.bin',
    'tencent_embedding_100k': 'word_vectors_100k.bin',
    'poem': 'poem.txt',
    'weibo': 'weibo_senti_100k/weibo_senti_100k.csv'
}


def get_data_path(dataset_name):
    base_path = get_base_dir()
    dataset_dir = base_path / 'datasets/storage'
    return dataset_dir / dataset_urls[dataset_name]
