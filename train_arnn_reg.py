import click

from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import *
from DatasetManager.the_session.folk_dataset import FolkDataset

from AnticipationRNN.anticipation_rnn_gauss_reg_model import *
from AnticipationRNN.anticipation_rnn_trainer import *
from AnticipationRNN.anticipation_rnn_tester import AnticipationRNNTester


@click.command()
@click.option('--note_embedding_dim', default=10,
              help='size of the note embeddings')
@click.option('--metadata_embedding_dim', default=2,
              help='size of the metadata embeddings')
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--lstm_hidden_size', default=256,
              help='hidden size of the LSTMs')
@click.option('--dropout_lstm', default=0.2,
              help='amount of dropout between LSTM layers')
@click.option('--input_dropout', default=0.2,
              help='amount of dropout between LSTM layers')
@click.option('--linear_hidden_size', default=256,
              help='hidden size of the Linear layers')
@click.option('--batch_size', default=32,
              help='training batch size')
@click.option('--num_epochs', default=50,
              help='number of training epochs')
@click.option('--train/--test', default=True,
              help='train or retrain the specified model')
@click.option('--log/--no_log', default=True,
              help='log the results for tensorboard')
@click.option('--lr', default=1e-4,
              help='learning rate')
@click.option('--plot/--no_plot', default=True,
              help='plot the training log')
@click.option('--teacher_forcing/--no_teacher_forcing', default=True,
              help='select if the model should use teacher forcing for training')
@click.option('--early_stop/--no_early_stop', default=True,
              help='select if early stopping is to be used')
def main(note_embedding_dim,
         metadata_embedding_dim,
         num_layers,
         lstm_hidden_size,
         dropout_lstm,
         input_dropout,
         linear_hidden_size,
         batch_size,
         num_epochs,
         train,
         log,
         lr,
         plot,
         teacher_forcing,
         early_stop
         ):
    # init dataset
    dataset_manager = DatasetManager()
    metadatas = [
        BeatMarkerMetadata(subdivision=6),
        TickMetadata(subdivision=6)
    ]
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
    folk_dataset: FolkDataset = dataset_manager.get_dataset(
        name='folk_4by4nbars_train',
        **folk_train_kwargs
    )
    folk_dataset_test: FolkDataset = dataset_manager.get_dataset(
        name='folk_4by4nbars_train',
        **folk_test_kwargs
    )

    model = ConstraintModelGaussianReg(
        dataset=folk_dataset,
        note_embedding_dim=note_embedding_dim,
        metadata_embedding_dim=metadata_embedding_dim,
        num_layers=num_layers,
        num_lstm_constraints_units=lstm_hidden_size,
        num_lstm_generation_units=lstm_hidden_size,
        linear_hidden_size=linear_hidden_size,
        dropout_prob=dropout_lstm,
        dropout_input_prob=input_dropout,
        unary_constraint=True,
        teacher_forcing=teacher_forcing
    )

    if train:
        if torch.cuda.is_available():
            model.cuda()
        trainer = AnticipationRNNGaussianRegTrainer(
            dataset=folk_dataset,
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
    tester = AnticipationRNNTester(
        dataset=folk_dataset_test,
        model=model
    )
    tester.test_model(
        batch_size=512
    )
    # gen_score, _, original_score = tester.generation_test()
    # gen_score.show()
    # original_score.show()

    # gen_score, _, original_score = tester.generation(tensor_score=None, start_measure=8, num_measures_gen=2)
    # gen_score.show()
    # original_score.show()


if __name__ == '__main__':
    main()
