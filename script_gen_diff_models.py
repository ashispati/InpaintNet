import click
import random

from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.the_session.folk_dataset import FolkDataset
from DatasetManager.metadata import TickMetadata, \
    BeatMarkerMetadata
from LatentRNN.latent_rnn import *
from LatentRNN.latent_rnn_tester import *
from LatentRNN.latent_rnn_trainer import *
from AnticipationRNN.anticipation_rnn_gauss_reg_model import *
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

    random.seed(0)

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

    # Initialize stuff
    test_filenames = folk_dataset_test.dataset_filenames
    num_melodies = 32
    num_measures = 16
    req_length = num_measures * 4 * 6
    num_past = 6
    num_future = 6
    num_target = 4
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    save_folder = 'saved_midi/'

    # First save original data
    for i in tqdm(range(num_melodies)):
        f = test_filenames[i]
        f_id = f[:-4]
        # save original scores
        save_filename = os.path.join(cur_dir, save_folder + f_id + '_original.mid')
        if os.path.isfile(save_filename):
            continue
        f = os.path.join(folk_dataset_test.corpus_it_gen.raw_dataset_dir, f)
        score = folk_dataset_test.corpus_it_gen.get_score_from_path(f, fix_and_expand=True)
        score_tensor = folk_dataset_test.get_score_tensor(score)
        metadata_tensor = folk_dataset_test.get_metadata_tensor(score)
        # ignore scores with less than 16 measures
        if score_tensor.size(1) < req_length:
            continue
        score_tensor = score_tensor[:, :req_length]
        metadata_tensor = metadata_tensor[:, :req_length, :]
        trunc_score = folk_dataset_test.tensor_to_score(score_tensor)
        trunc_score.write('midi', fp=save_filename)

    # Initialize models and testers
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
    latent_rnn_model.load()  # Latent RNN model must be pre-trained
    if torch.cuda.is_available():
        latent_rnn_model.cuda()
    latent_rnn_tester = LatentRNNTester(
        dataset=folk_dataset_test,
        model=latent_rnn_model
    )

    def process_latent_rnn_batch(score_tensor, num_past=6, num_future=6, num_target=4):
        assert(num_past + num_future + num_target == 16)
        score_tensor = score_tensor.unsqueeze(0)
        score_tensor = LatentRNNTrainer.split_to_measures(score_tensor, 24)
        tensor_past, tensor_future, tensor_target = LatentRNNTrainer.split_score(
            score_tensor=score_tensor,
            num_past=num_past,
            num_future=num_future,
            num_target=num_target,
            measure_seq_len=24
        )
        return tensor_past, tensor_future, tensor_target

    # Second save latent_rnn generations
    for i in tqdm(range(num_melodies)):
        f = test_filenames[i]
        f_id = f[:-4]
        save_filename = os.path.join(cur_dir, save_folder + f_id + '_latent_rnn.mid')
        if os.path.isfile(save_filename):
            continue
        f = os.path.join(folk_dataset_test.corpus_it_gen.raw_dataset_dir, f)
        score = folk_dataset_test.corpus_it_gen.get_score_from_path(f, fix_and_expand=True)
        score_tensor = folk_dataset_test.get_score_tensor(score)
        # metadata_tensor = folk_dataset_test.get_metadata_tensor(score)
        # ignore scores with less than 16 measures
        if score_tensor.size(1) < req_length:
            continue
        score_tensor = score_tensor[:, :req_length]
        # metadata_tensor = metadata_tensor[:, :req_length, :]
        # save regeneration using latent_rnn
        tensor_past, tensor_future, tensor_target = process_latent_rnn_batch(score_tensor, num_past, num_future, num_target)
        # forward pass through latent_rnn
        weights, gen_target, _ = latent_rnn_tester.model(
            past_context=tensor_past,
            future_context=tensor_future,
            target=tensor_target,
            measures_to_generate=num_target,
            train=False,
        )
        # convert to score
        batch_size, _, _ = gen_target.size()
        gen_target = gen_target.view(batch_size, num_target, 24)
        gen_score_tensor = torch.cat((tensor_past, gen_target, tensor_future), 1)
        latent_rnn_score = folk_dataset_test.tensor_to_score(gen_score_tensor.cpu())
        latent_rnn_score.write('midi', fp=save_filename)

    # Intialize arnn model and arnn_tester
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

    def process_arnn_batch(score_tensor, metadata_tensor, arnn_tester, num_past=6, num_target=4):
        score_tensor = score_tensor.unsqueeze(0)
        metadata_tensor = metadata_tensor.unsqueeze(0)
        tensor_score = to_cuda_variable_long(score_tensor)
        tensor_metadata = to_cuda_variable_long(metadata_tensor)
        constraints_location, start_tick, end_tick = arnn_tester.get_constraints_location(
            tensor_score,
            is_stochastic=False,
            start_measure=num_past,
            num_measures=num_target
        )
        arnn_batch = (tensor_score, tensor_metadata, constraints_location, start_tick, end_tick)
        return arnn_batch

    # Third save ARNN-Reg generations
    for i in tqdm(range(num_melodies)):
        f = test_filenames[i]
        f_id = f[:-4]
        save_filename = os.path.join(cur_dir, save_folder + f_id + '_arnn_reg.mid')
        if os.path.isfile(save_filename):
            continue
        f = os.path.join(folk_dataset_test.corpus_it_gen.raw_dataset_dir, f)
        score = folk_dataset_test.corpus_it_gen.get_score_from_path(f, fix_and_expand=True)
        score_tensor = folk_dataset_test.get_score_tensor(score)
        metadata_tensor = folk_dataset_test.get_metadata_tensor(score)
        # ignore scores with less than 16 measures
        if score_tensor.size(1) < req_length:
            continue
        score_tensor = score_tensor[:, :req_length]
        metadata_tensor = metadata_tensor[:, :req_length, :]
        # save regeneration using latent_rnn
        tensor_score, tensor_metadata, constraints_location, start_tick, end_tick = \
            process_arnn_batch(score_tensor, metadata_tensor, arnn_tester, num_past, num_target)
        # forward pass through latent_rnn
        _, gen_target = arnn_tester.model.forward_inpaint(
            score_tensor=tensor_score,
            metadata_tensor=tensor_metadata,
            constraints_loc=constraints_location,
            start_tick=start_tick,
            end_tick=end_tick,
        )
        # convert to score
        arnn_score = folk_dataset_test.tensor_to_score(gen_target.cpu())
        arnn_score.write('midi', fp=save_filename)

    # Intialize arnn-baseline model and arnn_tester
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
    # Fourth save ARNN-Baseline generations
    for i in tqdm(range(num_melodies)):
        f = test_filenames[i]
        f_id = f[:-4]
        save_filename = os.path.join(cur_dir, save_folder + f_id + '_arnn_baseline.mid')
        if os.path.isfile(save_filename):
            continue
        f = os.path.join(folk_dataset_test.corpus_it_gen.raw_dataset_dir, f)
        score = folk_dataset_test.corpus_it_gen.get_score_from_path(f, fix_and_expand=True)
        score_tensor = folk_dataset_test.get_score_tensor(score)
        metadata_tensor = folk_dataset_test.get_metadata_tensor(score)
        # ignore scores with less than 16 measures
        if score_tensor.size(1) < req_length:
            continue
        score_tensor = score_tensor[:, :req_length]
        metadata_tensor = metadata_tensor[:, :req_length, :]
        # save regeneration using latent_rnn
        tensor_score, tensor_metadata, constraints_location, start_tick, end_tick = \
            process_arnn_batch(score_tensor, metadata_tensor, arnn_baseline_tester, num_past, num_target)
        # forward pass through latent_rnn
        _, gen_target = arnn_baseline_tester.model.forward_inpaint(
            score_tensor=tensor_score,
            metadata_tensor=tensor_metadata,
            constraints_loc=constraints_location,
            start_tick=start_tick,
            end_tick=end_tick,
        )
        # convert to score
        arnn_baseline_score = folk_dataset_test.tensor_to_score(gen_target.cpu())
        arnn_baseline_score.write('midi', fp=save_filename)


if __name__ == '__main__':
    main()
