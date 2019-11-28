from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(dataset_name):
    base_path = Path(__file__).resolve().parent.parent
    dataset_dir = base_path / 'storage'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data, info = tfds.load(name=dataset_name, data_dir=dataset_dir, with_info=True, shuffle_files=True)
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
