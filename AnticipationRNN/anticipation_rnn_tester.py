import numpy as np
from random import randint

from AnticipationRNN.anticipation_rnn_gauss_reg_model import ConstraintModelGaussianReg
from AnticipationRNN.anticipation_rnn_trainer import AnticipationRNNGaussianRegTrainer
from utils.trainer import *


class AnticipationRNNTester(object):
    def __init__(self, dataset, model: ConstraintModelGaussianReg):
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.filepath = os.path.join('models/',
                                     self.model.__repr__())
        self.measure_seq_len = self.dataset.subdivision * self.dataset.num_beats_per_bar
        self.batch_size = 1
        self.measure_seq_len = 24

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
        # )
        print('Num Test Batches: ', len(gen_test))
        mean_loss_test, mean_accuracy_test = self.loss_and_acc_test(gen_test)
        print(f'Test Epoch: {1}/{1}')
        print(
            f'\tTest Loss: {mean_loss_test}'
            f'\tTest Accuracy: {mean_accuracy_test * 100} %'
        )

    def loss_and_acc_test(self, data_loader):
        """
        Computes loss and accuracy for test data (based on measures inpainting)
        :param data_loader: torch data loader object
        :return: (float, float)
        """
        mean_loss = 0
        mean_accuracy = 0

        for sample_id, batch in tqdm(enumerate(data_loader)):
            # process batch data
            batch_data = self.process_batch_data(batch)

            # extract data
            score_tensor, metadata_tensor, constraints_loc, start_tick, end_tick = batch_data

            # perform forward pass of model
            weights, activations = self.model.forward_inpaint(
                score_tensor=score_tensor,
                metadata_tensor=metadata_tensor,
                constraints_loc=constraints_loc,
                start_tick=start_tick,
                end_tick=end_tick,
            )
            targets = score_tensor[:, :, (constraints_loc[0, 0, :] == 0).nonzero().squeeze()]
            targets = targets.transpose(0, 1)
            # weights = [weight_per_voice[:, start_tick:end_tick, :] for weight_per_voice in weights]
            loss = self.mean_crossentropy_loss(
                weights=weights,
                targets=targets
            )
            accuracy = self.mean_accuracy(
                weights=weights,
                targets=targets
            )
            mean_loss += to_numpy(loss)
            mean_accuracy += to_numpy(accuracy)
        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    def loss_and_acc_test_alt(self, data_loader):
        """
        Computes loss and accuracy for test data (based on the training objective)
        :param data_loader: torch data loader object
        :return: (float, float)
        """
        mean_loss = 0
        mean_accuracy = 0

        for sample_id, batch in tqdm(enumerate(data_loader)):
            # process batch data
            batch_data = self.process_batch_data(batch)
            score_tensor, metadata_tensor, _, _, _ = batch_data
            weights, activations = self.model(
                score_tensor=score_tensor,
                metadata_tensor=metadata_tensor
            )

            t = int(
                (self.dataset.seq_size_in_beats * self.dataset.subdivision / 2)
            ) + np.random.randint(-5, 5)

            targets = score_tensor[:, :, t]
            targets = targets.transpose(0, 1)
            # targets is now (num_voices, batch)
            weights = [weight_per_voice[:, t, :] for weight_per_voice in weights]
            # list of (batch, num_notes)
            loss = AnticipationRNNGaussianRegTrainer.mean_crossentropy_loss(
                weights=weights,
                targets=targets
            )
            accuracy = AnticipationRNNGaussianRegTrainer.mean_accuracy(
                weights=weights,
                targets=targets
            )
            mean_loss += to_numpy(loss)
            mean_accuracy += to_numpy(accuracy)
        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    def generation_test(self):
        """
        Inpainting on a random sample from the test set
        ::return gen_score: music21 score object, the generated score
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
        tensor_score, tensor_metadata = self.process_batch_data(batch)
        batch_size, num_voices, seq_len, num_metadata = tensor_metadata.size()
        assert batch_size == 1
        tensor_score = tensor_score.view(num_voices, seq_len)
        tensor_metadata = tensor_metadata.view(num_voices, seq_len, num_metadata)
        constraints_location = torch.zeros_like(tensor_score)
        measure_seq_len = self.dataset.subdivision * self.dataset.num_beats_per_bar
        start_measure = 8  # TODO: remove this hard-coding
        num_measures_gen = 2  # TODO: remove this hard-coding
        start_tick = (start_measure - 1) * measure_seq_len
        end_tick = start_tick + num_measures_gen * measure_seq_len
        if start_tick > 0:
            constraints_location[:, :start_tick] = 1
        if end_tick < constraints_location.size(1) - 1:
            constraints_location[:, end_tick:] = 1
        tensor_past = tensor_score[:, :start_tick]
        tensor_future = tensor_score[:, end_tick:]
        tensor_target = tensor_score[:, start_tick:end_tick]

        # perform generation by forward pass through model
        _, gen_target, _ = self.model.generate(
            tensor_score=tensor_score,
            tensor_metadata=tensor_metadata,
            constraints_location=constraints_location,
            temperature=1.5)

        # concatenate past and future contexts
        gen_target = to_cuda_variable(gen_target[:, start_tick:end_tick])
        gen_score_tensor = torch.cat((tensor_past, gen_target, tensor_future), 1)
        gen_score = self.dataset.tensor_to_score(gen_score_tensor.cpu())
        # gen_score.show()
        original_tensor = torch.cat((tensor_past, tensor_target, tensor_future), 1)
        original_score = self.dataset.tensor_to_score(original_tensor.cpu())
        # original_score.show()
        return gen_score, gen_score_tensor, original_score

    def generation(self, tensor_score, start_measure, num_measures_gen):
        """
        Generates the measures using the measure RNN model
        :param tensor_score: torch Variable,
        :param start_measure: int, index of the 1st measure to be generated
                            must be >= 1
        :param num_measures_gen: int, number of measures to be generated
                            must be >= 1
        :return:
        """
        if tensor_score is None:
            score_gen = self.dataset.iterator_gen().__iter__()
            for _ in range(randint(0, 100)):
                original_score = next(score_gen)
        else:
            original_score = self.dataset.tensor_to_score(tensor_score)
        trans_interval = self.dataset.get_transpostion_interval_from_semitone(0)
        (tensor_score,
         tensor_metadata) = self.dataset.transposed_score_and_metadata_tensors(
            original_score,
            trans_interval
        )

        # prepare data
        if tensor_score.size(1) % self.measure_seq_len != 0:
            num_measures = int(np.floor(tensor_score.size(1) / self.measure_seq_len))
            tensor_score = tensor_score[:,  num_measures * self.measure_seq_len]
            tensor_metadata = tensor_metadata[:, num_measures * self.measure_seq_len]
        num_measures = min(16, tensor_score.size(1))
        tensor_score = tensor_score[:, :num_measures * self.measure_seq_len]
        tensor_metadata = tensor_metadata[:, :num_measures * self.measure_seq_len]
        constraints_location = torch.zeros_like(tensor_score)
        measure_seq_len = self.dataset.subdivision * self.dataset.num_beats_per_bar
        start_tick = (start_measure-1) * measure_seq_len
        end_tick = start_tick + num_measures_gen * measure_seq_len
        if start_tick > 0:
            constraints_location[:, :start_tick] = 1
        if end_tick < constraints_location.size(1) - 1:
            constraints_location[:, end_tick:] = 1
        tensor_past = tensor_score[:, :start_tick]
        tensor_future = tensor_score[:, end_tick:]
        tensor_target = tensor_score[:, start_tick:end_tick]

        # perform generation by forward pass through model
        _, gen_target, _ = self.model.generate(
            tensor_score=tensor_score,
            tensor_metadata=tensor_metadata,
            constraints_location=constraints_location,
            temperature=1.5)

        # concatenate past and future contexts
        gen_target = gen_target[:, start_tick:end_tick]
        gen_score_tensor = torch.cat((tensor_past, gen_target, tensor_future), 1)
        gen_score = self.dataset.tensor_to_score(gen_score_tensor)
        #gen_score.show()
        original_tensor = torch.cat((tensor_past, tensor_target, tensor_future), 1)
        original_score = self.dataset.tensor_to_score(original_tensor.cpu())
        #original_score.show()
        return gen_score, gen_score_tensor, original_score

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return:
        """
        tensor_score, tensor_metadata = batch
        # convert input to torch Variables
        tensor_score = to_cuda_variable_long(tensor_score)
        tensor_metadata = to_cuda_variable_long(tensor_metadata)

        constraints_location, start_tick, end_tick = self.get_constraints_location(
            tensor_score, is_stochastic=False
        )

        return tensor_score, tensor_metadata, constraints_location, start_tick, end_tick

    def get_constraints_location(self, tensor_score, is_stochastic, start_measure=None, num_measures=None):
        """
        Computes the constraint location
        :param tensor_score: torch Variable
                (batch_size, num_voices, seq_len)
        :param is_stochastic: bool, performs a stochastic split if TRUE
        :param start_measure: int, index of the starting measure
        :param num_measures: int, number of measures
        :return: constraint location, torch Variable,
                same shape as tensor_score, 0 if constraint active, 1 if inactive
        :return start_tick: int, index corresponding to the start of the constraints
        :return end_tick: int, index corresponding to the end of the constraints
        """
        constraints_location = torch.zeros_like(tensor_score)
        measure_seq_len = self.dataset.subdivision * self.dataset.num_beats_per_bar

        if is_stochastic:
            min_num_measures_past = 5
            min_num_measures_future = 5
            min_num_measures_target = 2
            num_measures = int(tensor_score.size(2) / measure_seq_len)
            assert (num_measures == self.dataset.n_bars)

            num_target = int(
                torch.randint(
                    low=min_num_measures_target,
                    high=num_measures - min_num_measures_past - min_num_measures_future,
                    size=(1,)
                ).item()
            )
            num_past = int(
                torch.randint(
                    low=min_num_measures_past,
                    high=num_measures - num_target - min_num_measures_future,
                    size=(1,)
                ).item()
            )
            num_future = num_measures - num_past - num_target
            assert (num_future >= min_num_measures_future)

            start_measure = num_past + 1
            num_measures = num_target
        else:
            if start_measure is None:
                start_measure = 8  # TODO: remove this hardcoding
            if num_measures is None:
                num_measures = 2  # TODO: remove this hardcoding

        start_tick = start_measure * measure_seq_len
        end_tick = start_tick + num_measures * measure_seq_len
        if start_tick > 0:
            constraints_location[:, :, :start_tick] = 1
        if end_tick < constraints_location.size(2) - 1:
            constraints_location[:, :, end_tick:] = 1
        return constraints_location, start_tick, end_tick

    @staticmethod
    def mean_crossentropy_loss(weights, targets):
        """
        Computes the average cross entropy loss
        :param weights: list of torch Variables, each of size
                (batch_size, seq_len, num_notes)
        :param targets:torch Variable,
                (num_voices, batch_size, seq_len)
        :return: loss, scalar
        """
        cross_entropy = nn.CrossEntropyLoss(size_average=True)
        _, batch_size, seq_len = targets.size()
        sum = 0
        for i, weight in enumerate(weights):
            w = weight.contiguous().view(batch_size * seq_len, -1)
            t = targets[i, :, :].contiguous().view(-1)
            ce = cross_entropy(w, t)
            sum += ce
        return sum / len(weights)

    @staticmethod
    def mean_accuracy(weights, targets):
        '''
        Computes the average accuracy
        :param weights: list of torch Variables, each of size
                (batch_size, seq_len, num_notes)
        :param targets:torch Variable,
                (num_voices, batch_size, seq_len)
        :return:
        '''
        sum = 0
        _, batch_size, seq_len = targets.size()
        for i, weight in enumerate(weights):
            w = weight.contiguous().view(batch_size * seq_len, -1)
            t = targets[i, :, :].contiguous().view(-1)
            max_values, max_indices = w.max(1)
            correct = max_indices == t
            sum += correct.float().mean()
        return sum / len(weights)
