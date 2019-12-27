import os
from tqdm import tqdm
from random import randint

from DatasetManager.the_session.folk_dataset import FolkDatasetNBars
from DatasetManager.helpers import START_SYMBOL, END_SYMBOL
from LatentRNN.latent_rnn import LatentRNN
from LatentRNN.latent_rnn_trainer import *
from utils.helpers import *
from utils.trainer import *


class LatentRNNTester(object):
    def __init__(self, dataset, model: LatentRNN):
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.filepath = os.path.join('models/',
                                     self.model.__repr__())
        self.min_num_measures_target = 1
        self.max_num_measure_target = 4
        assert (self.max_num_measure_target >= self.min_num_measures_target)
        assert (self.dataset.n_bars > self.min_num_measures_target)
        assert (self.dataset.n_bars > self.max_num_measure_target)
        self.measure_seq_len = self.dataset.subdivision * self.dataset.num_beats_per_bar
        self.batch_size = 1

    def test_model(self, batch_size=512):
        """
        Runs the model on the test set
        :return:
        """
        (_, gen_val, gen_test) = self.dataset.data_loaders(
            batch_size=batch_size,  # TODO: remove this hard coding
            split=(0.01, 0.01)
        )
        # print('Num Val Batches: ', len(gen_val))
        # mean_loss_val, mean_accuracy_val = self.loss_and_acc_test(gen_val)
        # print(f'Val Epoch: {1}/{1}')
        # print(
        #    f'\tTest Loss: {mean_loss_val}'
        #    f'\tTest Accuracy: {mean_accuracy_val * 100} %'
        #)
        print('Num Test Batches: ', len(gen_test))
        mean_loss_test, mean_accuracy_test = self.loss_and_acc_test(gen_test)
        print(f'Test Epoch: {1}/{1}')
        print(
            f'\tTest Loss: {mean_loss_test}'
            f'\tTest Accuracy: {mean_accuracy_test * 100} %'
        )

    def generation_test(self):
        """
        Inpainting on a random sample from the test set
        :return gen_score: music21 score object, the generated score
        :return gen_tensor_score: torch Variable,
                (1, num_measures, measure_seq_len)
        :return original_score: music21 score object, the original score
        """
        (_, gen_val, gen_test) = self.dataset.data_loaders(
            batch_size=1,  # TODO: remove this hard coding
            split=(0.70, 0.20)
        )
        gen_it_test = gen_test.__iter__()
        for _ in range(randint(0, len(gen_test))):
            batch = next(gen_it_test)

        # prepare data
        tensor_past, tensor_future, tensor_target = self.process_batch_data(batch)
        # generate
        num_target = tensor_target.size(1)
        return self.generate(tensor_past, tensor_future, tensor_target, num_target)

    def generation_random(self, tensor_score, start_measure, num_measures_gen):
        """
        Generates the measures using the measure RNN model
        :param tensor_score: torch Variable,
        :param start_measure: int, index of the 1st measure to be generated
        :param num_measures_gen: int, number of measures to be generated
        :return gen_score: music21 score object, the generated score
        :return gen_tensor_score: torch Variable,
                (1, num_measures, measure_seq_len)
        :return original_score: music21 score object, the original score
        """
        if tensor_score is None:
            score_gen = self.dataset.iterator_gen().__iter__()
            for _ in range(randint(1, 100)):
                original_score = next(score_gen)
            trans_interval = self.dataset.get_transpostion_interval_from_semitone(1)
            tensor_score, _ = self.dataset.transposed_score_and_metadata_tensors(
                original_score,
                trans_interval
            )
            if tensor_score is None:
                trans_interval = self.dataset.get_transpostion_interval_from_semitone(0)
                tensor_score, _ = self.dataset.transposed_score_and_metadata_tensors(
                    original_score,
                    trans_interval
                )
        else:
            original_score = self.dataset.tensor_to_score(tensor_score)
            trans_interval = self.dataset.get_transpostion_interval_from_semitone(0)
            tensor_score, _ = self.dataset.transposed_score_and_metadata_tensors(
                original_score,
                trans_interval
            )

        # prepare data
        if tensor_score.size(1) % self.measure_seq_len != 0:
            num_measures = int(np.floor(tensor_score.size(1) / self.measure_seq_len))
            tensor_score = tensor_score[:, :num_measures * self.measure_seq_len]
        tensor_score = torch.unsqueeze(tensor_score, 1)
        measures_tensor = LatentRNNTrainer.split_to_measures(
            score_tensor=tensor_score,
            measure_seq_len=self.measure_seq_len
        )
        num_measures = min(16, measures_tensor.size(1))
        tensor_score = tensor_score[:, :, :num_measures * self.measure_seq_len]
        num_past = start_measure - 1
        num_target = num_measures_gen
        num_future = num_measures - num_past - num_target
        tensor_past, tensor_future, tensor_target = LatentRNNTrainer.split_score(
            score_tensor=tensor_score,
            num_past=num_past,
            num_future=num_future,
            num_target=num_target,
            measure_seq_len=self.measure_seq_len
        )
        return self.generate(tensor_past, tensor_future, tensor_target, num_target)

    def generation(self,
                   num_iterations=None,
                   sequence_length_ticks=384,
                   tensor_score=None,
                   time_index_range_ticks=None
                   ):
        self.model.eval()
        if tensor_score is None:
            score_gen = self.dataset.iterator_gen().__iter__()
            for _ in range(2):#(randint(0, 100)):
                original_score = next(score_gen)
            trans_interval = self.dataset.get_transpostion_interval_from_semitone(1)
            tensor_score, _ = self.dataset.transposed_score_and_metadata_tensors(
                original_score,
                trans_interval
            )
            if tensor_score is None:
                trans_interval = self.dataset.get_transpostion_interval_from_semitone(0)
                tensor_score, _ = self.dataset.transposed_score_and_metadata_tensors(
                    original_score,
                    trans_interval
                )
        else:
            sequence_length_ticks = tensor_score.size(1)

        if time_index_range_ticks is None:
            start_measure = 8
            num_measures_gen = 2
        else:
            a_ticks, b_ticks = time_index_range_ticks
            assert a_ticks < b_ticks # check that you have to generate at least a few ticks
            assert a_ticks % self.measure_seq_len == 0
            assert b_ticks % self.measure_seq_len == 0
            start_measure = int(a_ticks / self.measure_seq_len) + 1
            num_measures_gen = int((b_ticks - a_ticks) / self.measure_seq_len)
            if a_ticks <= 0 or b_ticks >= sequence_length_ticks:
                score = self.dataset.tensor_to_score(tensor_score.cpu())
                return score, tensor_score, None

        if len(tensor_score.size()) == 2:
            if tensor_score.size(1) % self.measure_seq_len != 0:
                num_measures = int(np.floor(tensor_score.size(1) / self.measure_seq_len))
                tensor_score = tensor_score[:, :num_measures * self.measure_seq_len]
            tensor_score = torch.unsqueeze(tensor_score, 1)
            measures_tensor = LatentRNNTrainer.split_to_measures(
                score_tensor=tensor_score,
                measure_seq_len=self.measure_seq_len
            )
            num_measures = min(16, measures_tensor.size(1))
            tensor_score = tensor_score[:, :, :num_measures * self.measure_seq_len]
        else:
            raise ValueError('Invalid shape of tensor score')
        num_past = start_measure - 1
        num_target = num_measures_gen
        num_future = num_measures - num_past - num_target
        tensor_past, tensor_future, tensor_target = LatentRNNTrainer.split_score(
            score_tensor=tensor_score,
            num_past=num_past,
            num_future=num_future,
            num_target=num_target,
            measure_seq_len=self.measure_seq_len
        )
        score, tensor_score, _ =  self.generate(tensor_past, tensor_future, tensor_target, num_target)
        tensor_score = tensor_score.view(1, -1)
        return score, tensor_score, None

    def generate(self, tensor_past, tensor_future, tensor_target, num_target_measures, eval=False):
        """

        :param tensor_past: torch Variable, can be None
                (1, num_past_measures, measure_seq_len)
        :param tensor_future: torch Variable, can be None
                (1, num_future_measures, measure_seq_len)
        :param tensor_target: torch Variable, can be None
                (1, num_target_measures, measure_seq_len)
        :param num_target_measures: int, number of measures to generate,
                must be provided
        :param eval: bool, perform evaluation is True
        :return gen_score: music21 score object, the generated score
        :return gen_tensor_score: torch Variable,
                (1, num_measures, measure_seq_len)
        :return original_score: music21 score object, the original score
        """
        # sanity check
        if tensor_target is not None:
            if num_target_measures is not None:
                assert(num_target_measures == tensor_target.size(1))
            else:
                num_target_measures = tensor_target.size(1)
        else:
            if num_target_measures is None:
                raise ValueError

        # create dummy contexts if needed
        if tensor_past is None:
            tensor_past = self.create_empty_context('start')
        if tensor_future is None:
            tensor_future = self.create_empty_context('end')

        # perform forward pass of model
        weights, gen_target, _ = self.model(
            past_context=tensor_past,
            future_context=tensor_future,
            measures_to_generate=num_target_measures,
            train=False
        )

        # evaluate performance if target is available
        if tensor_target is not None and eval is True:
            # compute loss
            loss = Trainer.mean_crossentropy_loss_alt(
                weights=weights,
                targets=tensor_target
            )
            # compute accuracy
            accuracy = Trainer.mean_accuracy_alt(
                weights=weights,
                targets=tensor_target
            )
            print('Accuracy for Test Case:')
            print(
                f'\tLoss: {to_numpy(loss)}'
                f'\tAccuracy: {to_numpy(accuracy) * 100} %'
            )

        # convert to score
        batch_size, _, _ = gen_target.size()
        gen_target = gen_target.view(batch_size, num_target_measures, self.measure_seq_len)
        gen_score_tensor = torch.cat((tensor_past, gen_target, tensor_future), 1)
        gen_score = self.dataset.tensor_to_score(gen_score_tensor.cpu())
        if tensor_target is not None:
            original_tensor = torch.cat((tensor_past, tensor_target, tensor_future), 1)
            original_score = self.dataset.tensor_to_score(original_tensor.cpu())
        else:
            original_score = None
        return gen_score, gen_score_tensor, original_score

    def create_empty_context(self, type):
        """
        Creates an empty context tensor based on the type
        :param type: str, 'start' or 'end' or 'rest'
        :return: torch Variable,
                (1, num_measures, measure_seq_len)
        """
        if type == 'start':
            num_measures = 3
            symbol = self.dataset.note2index_dicts[self.dataset.NOTES][START_SYMBOL]
            context = to_cuda_variable_long(
                torch.ones(1, num_measures, self.measure_seq_len) * symbol
            )
        elif type == 'end':
            num_measures = 1
            symbol = self.dataset.note2index_dicts[self.dataset.NOTES][END_SYMBOL]
            context = to_cuda_variable_long(
                torch.ones(1, num_measures, self.measure_seq_len) * symbol
            )
        elif type == 'rest':
            num_measures = 1
            symbol = self.dataset.note2index_dicts[self.dataset.NOTES]['rest']
            context = to_cuda_variable_long(
                torch.ones(1, num_measures, self.measure_seq_len) * symbol
            )
        else:
            raise ValueError('Invalid argument "type"')
        return context

    def loss_and_acc_test(self, data_loader):
        """
         Computes loss and accuracy for test data
        :param data_loader: torch data loader object
        :return: (float, float)
        """
        mean_loss = 0
        mean_accuracy = 0

        for sample_id, batch in tqdm(enumerate(data_loader)):
            # process batch data
            batch_data = self.process_batch_data(batch)

            # extract data
            score_past, score_future, score_target = batch_data
            num_measures_past = score_past.size(1)
            num_measures_future = score_future.size(1)
            # perform forward pass of model
            weights, _, _ = self.model(
                past_context=score_past,
                future_context=score_future,
                target=score_target,
                measures_to_generate=self.dataset.n_bars - num_measures_past - num_measures_future,
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
            mean_loss += to_numpy(loss)
            mean_accuracy += to_numpy(accuracy)

        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        score_tensor, _ = batch
        batch_data = self.split_score_stochastic(score_tensor)
        # batch_data = LatentRNNTrainer.split_score(
        #    score_tensor=score_tensor,
        #    num_past=7,
        #    num_future=7,
        #    num_target=2,
        #    measure_seq_len=self.measure_seq_len
        # )
        return batch_data

    def split_score_stochastic(self, score_tensor, extra_outs=False, fix_num_target=None):
        """
        Splits the score tensor into past context, future context and target
        :param score_tensor: torch Tensor,
                (batch_size, 1, seq_len)
        :param extra_outs: bool, output additional items if True
        :param fix_num_target: int, fixes the number of target measures
        :return: tuple of 3 torch Variable objects
                score_past: torch Variable, past context
                            (batch_size, num_measures_past, measure_seq_len)
                score_future: torch Variable, future context
                            (batch_size, num_measures_future, measure_seq_len)
                score_target: torch Variable, measures to be predicted
                            (batch_size, num_of_target_measures * measure_seq_len)
        """
        measures_tensor = LatentRNNTrainer.split_to_measures(
            score_tensor=score_tensor,
            measure_seq_len=self.measure_seq_len
        )
        num_measures = measures_tensor.size(1)
        assert (num_measures == self.dataset.n_bars)
        if fix_num_target is None:
            num_target = int(
                torch.randint(
                    low=self.min_num_measures_target,
                    high=self.max_num_measure_target + 1,
                    # high=num_measures - self.min_num_measures_past - self.min_num_measures_future,
                    size=(1,)
                ).item()
            )
        else:
            num_target = fix_num_target
        num_past = int(
            torch.randint(
                # low=self.min_num_measures_past,
                # high=num_measures - num_target - self.min_num_measures_future,
                low=1,
                high=num_measures - num_target - 1,
                size=(1,)
            ).item()
        )
        num_future = num_measures - num_past - num_target
        # assert (num_future >= self.min_num_measures_future)

        tensor_past, tensor_future, tensor_target = LatentRNNTrainer.split_score(
            score_tensor=score_tensor,
            num_past=num_past,
            num_future=num_future,
            num_target=num_target,
            measure_seq_len=self.measure_seq_len
        )

        if extra_outs:
            return tensor_past, tensor_future, tensor_target, num_past, num_target
        else:
            return tensor_past, tensor_future, tensor_target
