import click

from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.the_session.folk_dataset import FolkDataset
from DatasetManager.metadata import TickMetadata, \
    BeatMarkerMetadata
from LatentRNN.latent_rnn_tester import LatentRNNTester
from AnticipationRNN.anticipation_rnn_tester import AnticipationRNNTester
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
@click.option('--batch_size', default=16,
              help='training batch size')
@click.option('--num_target', default=2,
              help='number of measures to generate')
@click.option('--num_models', default=4,
              help='number of models to test')
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
         num_layers,
         lstm_hidden_size,
         dropout_lstm,
         input_dropout,
         linear_hidden_size,
         batch_size,
         num_target,
         num_models
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
    if torch.cuda.is_available():
        vae_model.cuda()
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

    # init latent_rnn model and latent_rnn_tester
    latent_rnn_model = LatentRNN(
        dataset=folk_dataset_train,
        vae_model=vae_model,
        num_rnn_layers=num_latent_rnn_layers,
        rnn_hidden_size=latent_rnn_hidden_size,
        dropout=latent_rnn_dropout_prob,
        rnn_class=torch.nn.GRU,
        auto_reg=False,
        teacher_forcing=True
    )
    latent_rnn_model.load()  # latent_rnn model must be pre-trained
    if torch.cuda.is_available():
        latent_rnn_model.cuda()
    latent_rnn_tester = LatentRNNTester(
        dataset=folk_dataset_test,
        model=latent_rnn_model
    )

    # inti arnn model and arnn_testes
    arnn_model = ConstraintModelGaussianReg(
        dataset=folk_dataset_train,
        note_embedding_dim=note_embedding_dim,
        metadata_embedding_dim=metadata_embedding_dim,
        num_layers=num_layers,
        num_lstm_constraints_units=lstm_hidden_size,
        num_lstm_generation_units=lstm_hidden_size,
        linear_hidden_size=linear_hidden_size,
        dropout_prob=dropout_lstm,
        dropout_input_prob=input_dropout,
        unary_constraint=True,
        teacher_forcing=True
    )
    arnn_model.load()  # ARNN model must be pre-trained
    if torch.cuda.is_available():
        arnn_model.cuda()
    arnn_tester = AnticipationRNNTester(
        dataset=folk_dataset_test,
        model=arnn_model
    )

    arnn_baseline_model = AnticipationRNNBaseline(
        dataset=folk_dataset_train,
        note_embedding_dim=note_embedding_dim,
        metadata_embedding_dim=metadata_embedding_dim,
        num_layers=num_layers,
        num_lstm_constraints_units=lstm_hidden_size,
        num_lstm_generation_units=lstm_hidden_size,
        linear_hidden_size=linear_hidden_size,
        dropout_prob=dropout_lstm,
        dropout_input_prob=input_dropout,
        unary_constraint=True,
        teacher_forcing=True
    )
    arnn_baseline_model.load()  # ARNN model must be pre-trained
    if torch.cuda.is_available():
        arnn_baseline_model.cuda()
    arnn_baseline_tester = AnticipationRNNTester(
        dataset=folk_dataset_test,
        model=arnn_baseline_model
    )

    # create test dataloader
    (_, _, test_dataloader) = folk_dataset_test.data_loaders(
        batch_size=batch_size,
        split=(0.01, 0.01)
    )

    # test
    print('Num Test Batches: ', len(test_dataloader))
    latent_rnn_mean_loss, latent_rnn_mean_accuracy, \
    arnn_mean_loss, arnn_mean_accuracy, \
    arnn_baseline_mean_loss, arnn_baseline_mean_accuracy = loss_and_acc_test(
        data_loader=test_dataloader,
        latent_rnn_tester=latent_rnn_tester,
        arnn_tester=arnn_tester,
        arnn_baseline_tester=arnn_baseline_tester,
        num_target_measures=num_target,
        num_models=num_models
    )
    print('Test Epoch:')
    print(
        'latent_rnn Test Loss: ', latent_rnn_mean_loss, '\n'
        'latent_rnn Test Accuracy: ', latent_rnn_mean_accuracy * 100, '\n'
        'ARNN Test Loss: ', arnn_mean_loss, '\n'
        'ARNN Test Accuracy: ', arnn_mean_accuracy * 100, '\n'
        'ARNN Baseline Test Loss: ', arnn_baseline_mean_loss, '\n'
        'ARNN Baseline Test Accuracy: ', arnn_baseline_mean_accuracy * 100, '\n'
    )


def process_batch_data(batch, latent_rnn_tester, arnn_tester, num_target_measures=2):
    """
    Processes the batch returned by the dataloader iterator
    :param batch: object returned by the dataloader iterator
    :param latent_rnn_tester: LatentRNNTester object
    :param arnn_tester: AnticipationRNNTester object
    :param num_target_measures: int, number of measures to be generated
    :return:
    """
    score_tensor, metadata_tensor = batch
    score_past, score_future, score_target, num_past, num_target = \
        latent_rnn_tester.split_score_stochastic(
            score_tensor,
            extra_outs=True,
            fix_num_target=num_target_measures
        )
    # compute latent_rnn batch
    latent_rnn_batch = (score_past, score_future, score_target)
    # compute arnn batch
    tensor_score = to_cuda_variable_long(score_tensor)
    tensor_metadata = to_cuda_variable_long(metadata_tensor)
    constraints_location, start_tick, end_tick = arnn_tester.get_constraints_location(
        tensor_score,
        is_stochastic=False,
        start_measure=num_past,
        num_measures=num_target
    )
    arnn_batch = (tensor_score, tensor_metadata, constraints_location, start_tick, end_tick)
    return latent_rnn_batch, arnn_batch


def loss_and_acc_test(
        data_loader,
        latent_rnn_tester: LatentRNNTester,
        arnn_tester: AnticipationRNNTester,
        arnn_baseline_tester: AnticipationRNNTester = None,
        num_target_measures=2,
        num_models=4,
):
    """
    Computes loss and accuracy for test data (based on measures inpainting)
    :param data_loader: torch data loader object
    :param latent_rnn_tester: LatentRNNTester object
    :param arnn_tester: AnticipationRNNTester object
    :param arnn_baseline_tester: AnticipationRNNTester object
    :param num_target_measures: int, number of measures to be generated
    :param num_models: int, number of models to be tested
    :return: (float, float)
    """
    latent_rnn_mean_loss = 0
    latent_rnn_mean_accuracy = 0
    arnn_mean_loss = 0
    arnn_mean_accuracy = 0
    arnn_baseline_mean_loss = 0
    arnn_baseline_mean_accuracy = 0
    for sample_id, batch in tqdm(enumerate(data_loader)):
        # process batch data
        latent_rnn_batch, arnn_batch = process_batch_data(batch, latent_rnn_tester, arnn_tester, num_target_measures)

        # extract data
        score_past, score_future, score_target = latent_rnn_batch
        tensor_score, tensor_metadata, constraints_location, start_tick, end_tick = arnn_batch

        # ARNN model first
        if num_models >= 1:
            # perform forward pass
            weights, _ = arnn_tester.model.forward_inpaint(
                score_tensor=tensor_score,
                metadata_tensor=tensor_metadata,
                constraints_loc=constraints_location,
                start_tick=start_tick,
                end_tick=end_tick,
            )
            targets = tensor_score[:, :, start_tick:end_tick]
            targets = targets.transpose(0, 1)
            # compute loss
            loss = arnn_tester.mean_crossentropy_loss(
                weights=weights,
                targets=targets
            )
            # compute accuracy
            accuracy = arnn_tester.mean_accuracy(
                weights=weights,
                targets=targets
            )
            arnn_mean_loss += to_numpy(loss)
            arnn_mean_accuracy += to_numpy(accuracy)
        else:
            arnn_mean_loss += 0
            arnn_mean_accuracy += 0

        # latent_rnn model second
        if num_models >= 2:
            # perform forward pass
            num_measures_past = score_past.size(1)
            num_measures_future = score_future.size(1)
            target_num_measures = latent_rnn_tester.dataset.n_bars - num_measures_past - num_measures_future
            weights, _, _ = latent_rnn_tester.model.forward(
                past_context=score_past,
                future_context=score_future,
                target=score_target,
                measures_to_generate=target_num_measures,
                train=False,
            )
            # compute loss
            loss = Trainer.mean_crossentropy_loss_alt(
                weights=weights,
                targets=score_target
            )
            # compute accuracy
            accuracy = Trainer.mean_accuracy_alt(
                weights=weights,
                targets=score_target
            )
            latent_rnn_mean_loss += to_numpy(loss)
            latent_rnn_mean_accuracy += to_numpy(accuracy)
        else:
            latent_rnn_mean_loss += 0
            latent_rnn_mean_accuracy += 0

    latent_rnn_mean_loss /= len(data_loader)
    latent_rnn_mean_accuracy /= len(data_loader)
    arnn_mean_loss /= len(data_loader)
    arnn_mean_accuracy /= len(data_loader)
    arnn_baseline_mean_loss /= len(data_loader)
    arnn_baseline_mean_accuracy /= len(data_loader)
    return (
        latent_rnn_mean_loss,
        latent_rnn_mean_accuracy,
        arnn_mean_loss,
        arnn_mean_accuracy,
        arnn_baseline_mean_loss,
        arnn_baseline_mean_accuracy,
    )


if __name__ == '__main__':
    main()
