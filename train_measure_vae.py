import click

from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.the_session.folk_dataset import FolkDataset
from DatasetManager.metadata import \
    TickMetadata, \
    BeatMarkerMetadata
from MeasureVAE.measure_vae import MeasureVAE
from MeasureVAE.vae_trainer import VAETrainer
from MeasureVAE.vae_tester import VAETester
from utils.helpers import *


@click.command()
@click.option('--note_embedding_dim', default=10,
              help='size of the note embeddings')
@click.option('--metadata_embedding_dim', default=2,
              help='size of the metadata embeddings')
@click.option('--num_encoder_layers', default=2,
              help='number of layers in encoder RNN')
@click.option('--encoder_hidden_size', default=512,
              help='hidden size of the encoder RNN')
@click.option('--encoder_dropout_prob', default=0.5,
              help='float, amount of dropout prob between encoder RNN layers')
@click.option('--has_metadata', default=False,
              help='bool, True if data contains metadata')
@click.option('--latent_space_dim', default=256,
              help='int, dimension of latent space parameters')
@click.option('--num_decoder_layers', default=2,
              help='int, number of layers in decoder RNN')
@click.option('--decoder_hidden_size', default=512,
              help='int, hidden size of the decoder RNN')
@click.option('--decoder_dropout_prob', default=0.5,
              help='float, amount got dropout prob between decoder RNN layers')
@click.option('--batch_size', default=256,
              help='training batch size')
@click.option('--num_epochs', default=30,
              help='number of training epochs')
@click.option('--train/--test', default=True,
              help='train or retrain the specified model')
@click.option('--plot/--no_plot', default=False,
              help='plot the training log')
@click.option('--log/--no_log', default=True,
              help='log the results for tensorboard')
@click.option('--lr', default=1e-4,
              help='learning rate')
def main(note_embedding_dim,
         metadata_embedding_dim,
         num_encoder_layers,
         encoder_hidden_size,
         encoder_dropout_prob,
         latent_space_dim,
         num_decoder_layers,
         decoder_hidden_size,
         decoder_dropout_prob,
         has_metadata,
         batch_size,
         num_epochs,
         train,
         plot,
         log,
         lr
         ):

    dataset_manager = DatasetManager()
    metadatas = [
        BeatMarkerMetadata(subdivision=6),
        TickMetadata(subdivision=6)
    ]
    mvae_train_kwargs = {
        'metadatas': metadatas,
        'sequences_size': 32,
        'num_bars': 16,
        'train': True
    }
    mvae_test_kwargs = {
        'metadatas': metadatas,
        'sequences_size': 32,
        'num_bars': 16,
        'train': False
    }
    folk_dataset: FolkDataset = dataset_manager.get_dataset(
        name='folk_4by4nbars_train',
        **mvae_train_kwargs
    )

    folk_dataset_test: FolkDataset = dataset_manager.get_dataset(
        name='folk_4by4nbars_train',
        **mvae_test_kwargs
    )

    model = MeasureVAE(
        dataset=folk_dataset,
        note_embedding_dim=note_embedding_dim,
        metadata_embedding_dim=metadata_embedding_dim,
        num_encoder_layers=num_encoder_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_dropout_prob=encoder_dropout_prob,
        latent_space_dim=latent_space_dim,
        num_decoder_layers=num_decoder_layers,
        decoder_hidden_size=decoder_hidden_size,
        decoder_dropout_prob=decoder_dropout_prob,
        has_metadata=has_metadata
    )

    if train:
        if torch.cuda.is_available():
            model.cuda()
        trainer = VAETrainer(
            dataset=folk_dataset,
            model=model,
            lr=lr
        )
        trainer.train_model(
            batch_size=batch_size,
            num_epochs=num_epochs,
            plot=plot,
            log=log
        )
    else:
        model.load()
        model.cuda()
        model.eval()

    tester = VAETester(
        dataset=folk_dataset_test,
        model=model
    )
    tester.test_model()


if __name__ == '__main__':
    main()
