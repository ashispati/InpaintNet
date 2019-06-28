import click

from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.the_session.folk_dataset import FolkDataset
from DatasetManager.metadata import FermataMetadata, \
    TickMetadata, \
    KeyMetadata, \
    BeatMarkerMetadata
from MeasureVAE.measure_vae import MeasureVAE
from LatentRNN.latent_rnn import LatentRNN
from LatentRNN.latent_rnn_trainer import LatentRNNTrainer
from LatentRNN.latent_rnn_tester import LatentRNNTester
from MeasureVAE.vae_tester import *
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
@click.option('--has_metadata', default=True,
              help='bool, True if data contains metadata')
@click.option('--latent_space_dim', default=256,
              help='int, dimension of latent space parameters')
@click.option('--num_decoder_layers', default=2,
              help='int, number of layers in decoder RNN')
@click.option('--decoder_hidden_size', default=512,
              help='int, hidden size of the decoder RNN')
@click.option('--decoder_dropout_prob', default=0.5,
              help='float, amount got dropout prob between decoder RNN layers')
@click.option('--num_latent_rnn_layers', default=2,
              help='number of layers in measure RNN')
@click.option('--latent_rnn_hidden_size', default=512,
              help='hidden size of the measure RNN')
@click.option('--latent_rnn_dropout_prob', default=0.5,
              help='float, amount of dropout prob between measure RNN layers')
@click.option('--batch_size', default=32,
              help='training batch size')
@click.option('--num_epochs', default=50,
              help='number of training epochs')
@click.option('--train/--test', default=True,
              help='train or retrain the specified model')
@click.option('--lr', default=1e-4,
              help='learning rate')
@click.option('--plot/--no_plot', default=True,
              help='plot the training log')
@click.option('--log/--no_log', default=True,
              help='log the results for tensorboard')
@click.option('--auto_reg/--no_auto_reg', default=True,
              help='select if the model should be auto-regressive')
@click.option('--teacher_forcing/--no_teacher_forcing', default=True,
              help='select if the model should use teacher forcing for training')
@click.option('--early_stop/--no_early_stop', default=True,
              help='select if early stopping is to be used')
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
         num_latent_rnn_layers,
         latent_rnn_hidden_size,
         latent_rnn_dropout_prob,
         batch_size,
         num_epochs,
         train,
         lr,
         plot,
         log,
         auto_reg,
         teacher_forcing,
         early_stop
         ):

    # init dataset
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
    folk_dataset_vae: FolkDataset = dataset_manager.get_dataset(
        name='folk_4by4nbars_train',
        **mvae_train_kwargs
    )
    # init vae model
    vae_model = MeasureVAE(
        dataset=folk_dataset_vae,
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
    vae_model.load()  # VAE model must be pre-trained

    folk_train_kwargs = {
        'metadatas': metadatas,
        'sequences_size': 32,
        'num_bars': 16,
        'train': True
    }
    folk_test_kwargs = {
        'metadatas': metadatas,
        'sequences_size': 32,
        'num_bars': 16,
        'train': False
    }
    folk_dataset_train: FolkDataset = dataset_manager.get_dataset(
        name='folk_4by4nbars_train',
        **folk_train_kwargs
    )
    folk_dataset_test: FolkDataset = dataset_manager.get_dataset(
        name='folk_4by4nbars_train',
        **folk_test_kwargs
    )

    # init latent_rnn model
    model = LatentRNN(
        dataset=folk_dataset_train,
        vae_model=vae_model,
        num_rnn_layers=num_latent_rnn_layers,
        rnn_hidden_size=latent_rnn_hidden_size,
        dropout=latent_rnn_dropout_prob,
        rnn_class=torch.nn.GRU,
        auto_reg=auto_reg,
        teacher_forcing=teacher_forcing
    )

    if train:
        if torch.cuda.is_available():
            model.cuda()
        trainer = LatentRNNTrainer(
            dataset=folk_dataset_train,
            model=model,
            lr=lr,
            early_stopping=early_stop
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
    tester = LatentRNNTester(
        dataset=folk_dataset_test,
        model=model
    )
    tester.test_model(
        batch_size=batch_size
    )
    '''
    gen_score, _, original_score = tester.generation_random(
        tensor_score=None,
        start_measure=8,
        num_measures_gen=2
    )
    gen_score.show()
    original_score.show()

    gen_score2, _, original_score2 = tester.generation_test()
    gen_score2.show()
    original_score2.show()
    '''


if __name__ == '__main__':
    main()
