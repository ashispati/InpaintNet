from DatasetManager.the_session.folk_dataset import *
from DatasetManager.the_session.folk_data_helpers \
             import FolkIteratorGenerator


all_datasets = {
    'folk':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=None,
                time_sigs=[(3, 4), (4, 4)]
            )
        },
    'folk_test':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=10,
                time_sigs=[(3, 4), (4, 4)]
            )
        },
    'folk_4by4_test':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=100,
                time_sigs=[(4, 4)]
            ) 
        },
    'folk_4by4':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=None,
                time_sigs=[(4, 4)]
            ) 
        },
    'folk_3by4_test':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=100,
                time_sigs=[(3, 4)]
            ) 
        },
    'folk_3by4':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=None,
                time_sigs=[(3, 4)]
            ) 
        },
    'folk_4by4measures_test':
        {
            'dataset_class_name': FolkMeasuresDataset,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=100,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4measures_test2':
        {
            'dataset_class_name': FolkMeasuresDataset,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=1,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4measures':
        {
            'dataset_class_name': FolkMeasuresDataset,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=None,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4measurestr_test':
        {
            'dataset_class_name': FolkMeasuresDatasetTranspose,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=1000,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4measurestr':
        {
            'dataset_class_name': FolkMeasuresDatasetTranspose,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=None,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4nbars_short':
        {
            'dataset_class_name': FolkDatasetNBars,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=10,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4nbars':
        {
            'dataset_class_name': FolkDatasetNBars,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=None,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4nbars_train':
        {
            'dataset_class_name': FolkDatasetNBars,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=None,
                time_sigs=[(4, 4)]
            )
        },
}


class DatasetManager:
    def __init__(self):
        self.package_dir = os.path.dirname(os.path.realpath(__file__))
        self.cache_dir = os.path.join(self.package_dir,
                                      'dataset_cache')
        # create cache dir if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

    def get_dataset(self, name: str, **dataset_kwargs) -> MusicDataset:
        if name in all_datasets:
            return self.load_if_exists_or_initialize_and_save(
                name=name,
                **all_datasets[name],
                **dataset_kwargs
            )
        else:
            print('Dataset with name {name} is not registered in all_datasets variable')
            raise ValueError

    def load_if_exists_or_initialize_and_save(
            self,
            dataset_class_name,
            corpus_it_gen,
            name,
            **kwargs
    ):
        """

        :param dataset_class_name:
        :param corpus_it_gen:
        :param name:
        :param kwargs: parameters specific to an implementation
        of MusicDataset
        :return:
        """
        kwargs.update(
            {'name':          name,
             'corpus_it_gen': corpus_it_gen,
             'cache_dir': self.cache_dir
             })
        dataset = dataset_class_name(**kwargs)
        if os.path.exists(dataset.filepath):
            print(f'Loading {dataset.__repr__()} from {dataset.filepath}')
            dataset = torch.load(dataset.filepath)
            new_cache_dir = kwargs['cache_dir']
            org_cache_dir = dataset.cache_dir
            dataset.cache_dir = new_cache_dir
            dataset.dict_path = dataset.dict_path.replace(org_cache_dir, new_cache_dir)
            dataset.dicts_dir = dataset.dicts_dir.replace(org_cache_dir, new_cache_dir)
            dataset.corpus_it_gen = corpus_it_gen
            print(f'(the corresponding TensorDataset is not loaded)')
        else:
            print(f'Creating {dataset.__repr__()}, '
                  f'both tensor dataset and parameters')
            # initialize and force the computation of the tensor_dataset
            # first remove the cached data if it exists
            if os.path.exists(dataset.tensor_dataset_filepath):
                os.remove(dataset.tensor_dataset_filepath)
            # recompute dataset parameters and tensor_dataset
            # this saves the tensor_dataset in dataset.tensor_dataset_filepath
            tensor_dataset = dataset.tensor_dataset
            # save all dataset parameters EXCEPT the tensor dataset
            # which is stored elsewhere
            dataset.tensor_dataset = None
            torch.save(dataset, dataset.filepath)
            print(f'{dataset.__repr__()} saved in {dataset.filepath}')
            dataset.tensor_dataset = tensor_dataset
        return dataset


# Usage example
if __name__ == '__main__':
    dataset_manager = DatasetManager()
    # Folk Dataset  
    metadatas = [
        BeatMarkerMetadata(subdivision=6),
        TickMetadata(subdivision=6)
    ]
    folk_dataset_kwargs = {
        'metadatas':        metadatas,
        'sequences_size':   32
    }
    folk_dataset: FolkDataset = dataset_manager.get_dataset(
        name='folk_4by4nbars',
        **folk_dataset_kwargs
    )
    (train_dataloader,
     val_dataloader,
     test_dataloader) = folk_dataset.data_loaders(
        batch_size=256,
        split=(0.7, 0.2)
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))
