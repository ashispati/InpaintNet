import torch
import random
import numpy as np
import os
from torch import nn
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding

from DatasetManager.music_dataset import MusicDataset
from utils.helpers import *
from utils.model import *


def lstm_with_activations(lstm_list, input, hidden, dropout_layer=None):
    x = input
    h, c = hidden
    all_hs = []
    all_last_h = []
    all_last_c = []
    num_layers = len(lstm_list)
    for layer_index, lstm in enumerate(lstm_list):
        h_i = h[layer_index: layer_index + 1]
        c_i = c[layer_index: layer_index + 1]
        hidden_i = h_i, c_i
        x, (last_h_i, last_c_i) = lstm(x, hidden_i)
        # TODO before or after
        # before is better
        if dropout_layer and layer_index < num_layers - 1:
            x = dropout_layer(x)

        all_last_h.append(last_h_i.unsqueeze(0))
        all_last_c.append(last_c_i.unsqueeze(0))
        all_hs.append(x.unsqueeze(0))

    all_hs = torch.cat(all_hs, 0)
    all_last_h = torch.cat(all_last_h, 0)
    all_last_c = torch.cat(all_last_c, 0)
    last_hidden = (all_last_h, all_last_c)
    return (x, last_hidden), all_hs


class ConstraintModelGaussianReg(Model):
    def __init__(self, dataset: MusicDataset,
                 note_embedding_dim=20,
                 metadata_embedding_dim=30,
                 num_lstm_constraints_units=256,
                 num_lstm_generation_units=256,
                 linear_hidden_size=128,
                 num_layers=1,
                 dropout_input_prob=0.2,
                 dropout_prob=0.5,
                 unary_constraint=False,
                 teacher_forcing=True
                 ):
        super(ConstraintModelGaussianReg, self).__init__()
        self.dataset = dataset
        self.use_teacher_forcing = teacher_forcing
        self.teacher_forcing_prob = 0.5

        # === parameters ===
        # --- common parameters
        self.num_layers = num_layers
        self.num_units_linear = linear_hidden_size
        self.unary_constraint = unary_constraint
        unary_constraint_size = 1 if self.unary_constraint else 0

        # --- notes
        self.note_embedding_dim = note_embedding_dim
        self.num_lstm_generation_units = num_lstm_generation_units
        self.num_notes_per_voice = [len(d)
                                    for d in self.dataset.note2index_dicts
                                    ]
        # use also note_embeddings to embed unary constraints
        self.note_embeddings = ModuleList(
            [
                Embedding(num_embeddings + unary_constraint_size, self.note_embedding_dim)
                for num_embeddings in self.num_notes_per_voice
            ]
        )
        # todo different ways of merging constraints

        # --- metadatas
        self.metadata_embedding_dim = metadata_embedding_dim
        self.num_elements_per_metadata = [metadata.num_values
                                          for metadata in self.dataset.metadatas]
        # must add the number of voices
        self.num_elements_per_metadata.append(self.dataset.num_voices)
        # embeddings for all metadata except unary constraints
        self.metadata_embeddings = ModuleList(
            [
                Embedding(num_embeddings, self.metadata_embedding_dim)
                for num_embeddings in self.num_elements_per_metadata
            ]
        )
        # nn hyper parameters
        self.num_lstm_constraints_units = num_lstm_constraints_units
        self.dropout_input_prob = dropout_input_prob
        self.dropout_prob = dropout_prob

        lstm_constraint_num_hidden = [
                                         (self.metadata_embedding_dim * len(
                                             self.num_elements_per_metadata)
                                          + self.note_embedding_dim * unary_constraint_size,
                                          self.num_lstm_constraints_units)
                                     ] + [(self.num_lstm_constraints_units,
                                           self.num_lstm_constraints_units)] * (
                                             self.num_layers - 1)
        # trainable parameters

        self.lstm_constraint = nn.ModuleList([nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout_prob,
            batch_first=True)
            for input_size, hidden_size in lstm_constraint_num_hidden
        ])
        lstm_generation_num_hidden = [
                                         (self.note_embedding_dim + self.num_lstm_constraints_units,
                                          self.num_lstm_constraints_units)
                                     ] + [(self.num_lstm_constraints_units,
                                           self.num_lstm_constraints_units)] * (
                                             self.num_layers - 1)

        self.lstm_generation = nn.ModuleList([nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout_prob,
            batch_first=True)
            for input_size, hidden_size in lstm_generation_num_hidden
        ]
        )
        self.linear_1 = nn.Linear(self.num_lstm_generation_units, linear_hidden_size)
        self.linear_ouput_notes = ModuleList(
            [
                nn.Linear(self.num_units_linear, num_notes)
                for num_notes in self.num_notes_per_voice
            ]
        )
        # todo test real dropout input
        self.dropout_layer = nn.Dropout2d(p=dropout_input_prob)
        self.dropout_lstm_layer = nn.Dropout(p=dropout_prob)

        self.optimizer = torch.optim.Adam(self.parameters())

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.filepath = os.path.join(cur_dir, 'models/',
                                     self.__repr__())

    def __repr__(self):
        filestr = f'AnticipationRNNReg(' \
               f'{self.dataset.__repr__()},' \
               f'{self.note_embedding_dim},' \
               f'{self.metadata_embedding_dim},' \
               f'{self.num_lstm_constraints_units},' \
               f'{self.num_lstm_generation_units},' \
               f'{self.num_units_linear},' \
               f'{self.num_layers},' \
               f'{self.dropout_input_prob},' \
               f'{self.dropout_prob},' \
               f'{self.unary_constraint},' \
               f')'
        if self.use_teacher_forcing:
            filestr += ',tf'
        else:
            filestr += ',no_tf'
        return filestr

    @staticmethod
    def flatten_tensor_score(chorale: Variable):
        """
        :param chorale:(batch, num_voices, length, embedding_dim)
        :return: (batch, num_voices * length) with num_voices varying faster
        """
        # todo make it independant of number of dimensions after the third one
        batch_size, num_voices, length, embedding_dim = chorale.size()
        chorale = chorale.transpose(1, 2).contiguous()
        chorale = chorale.view(batch_size, num_voices * length, embedding_dim)
        # todo check
        return chorale

    @staticmethod
    def flatten_metadata(metadata: Variable):
        batch_size, num_voices, length, num_metadatas = metadata.size()
        metadata = metadata.transpose(1, 2).contiguous()
        metadata = metadata.view(batch_size, num_voices * length, num_metadatas)
        return metadata

    def _forward_no_tf(self, score_tensor: Variable, metadata_tensor: Variable, constraints_loc):
        batch_size, num_voices, chorale_length = score_tensor.size()
        sequence_length = num_voices * chorale_length
        gen_chorale = self.dataset.empty_score_tensor(chorale_length).unsqueeze(0)
        gen_chorale = gen_chorale.repeat(batch_size, 1, 1)
        # === embed as wrapped sequence ===
        # --- chorale
        x = self.embed_tensor_score(score_tensor)

        # --- metadata
        m = self.embed_metadata(
            metadata_tensor,
            score_tensor,
            constraints_location=constraints_loc
        )
        # === LSTM on constraints ===
        output_constraints, activations_constraint = self.output_lstm_constraints(m)
        hidden = init_hidden_lstm(
            num_layers=self.num_layers,
            batch_size=batch_size,
            lstm_hidden_size=self.num_lstm_generation_units
        )

        final_weights = [[] for i in range(num_voices)]
        for tick_index in range(-1, chorale_length-1):
            voice_index = tick_index % num_voices
            time_index = (tick_index - voice_index) // num_voices
            next_voice_index = (tick_index + 1) % num_voices
            next_time_index = (tick_index + 1 - next_voice_index) // num_voices

            if tick_index == -1:
                last_start_symbol = 0  # gen_chorale[-1, 0]
                last_start_symbol = torch.from_numpy(np.array([last_start_symbol]))[None, :]
                time_slice = self.note_embeddings[voice_index](
                    to_cuda_variable(last_start_symbol)
                )
                time_slice = time_slice.repeat(batch_size, 1, 1)
            else:
                time_slice = gen_chorale[:, voice_index:voice_index + 1, time_index]
                # time_slice = torch.from_numpy(np.array([time_slice]))[None, :]
                note = self.note_embeddings[voice_index](
                    to_cuda_variable(time_slice)
                )
                time_slice = note

            time_slice_cat = torch.cat((time_slice, output_constraints[:, tick_index + 1:tick_index + 2, :]), 2)

            (output_gen, hidden), activations_generation = lstm_with_activations(
                lstm_list=self.lstm_generation,
                input=time_slice_cat, hidden=hidden)
            h, c = hidden
            hidden = h[:, 0, :, :], c[:, 0, :, :]

            weights = F.relu(self.linear_1(output_gen[:, 0, :]), inplace=True)
            weights = self.linear_ouput_notes[next_voice_index](weights)

            # compute predictions
            # temperature
            weights = weights
            preds = F.softmax(weights)
            final_weights[voice_index].append(weights.unsqueeze(1))

            # first batch element
            preds = to_numpy(preds[0])
            new_pitch_index = np.argmax(preds)

            gen_chorale[:, next_voice_index, next_time_index] = int(new_pitch_index)
        for i in range(num_voices):
            final_weights[i] = torch.cat(final_weights[i], 1)
        return final_weights, gen_chorale

    def forward_inpaint(self, score_tensor: Variable, metadata_tensor: Variable, constraints_loc, start_tick, end_tick):
        batch_size, num_voices, chorale_length = score_tensor.size()
        sequence_length = num_voices * chorale_length
        gen_chorale = self.dataset.empty_score_tensor(chorale_length).unsqueeze(0)
        gen_chorale = gen_chorale.repeat(batch_size, 1, 1)
        gen_chorale[:, :, :start_tick] = score_tensor[:, :, :start_tick]
        gen_chorale[:, :, end_tick:] = score_tensor[:, :, end_tick:]
        # === embed as wrapped sequence ===
        # --- chorale
        x = self.embed_tensor_score(score_tensor)

        # --- metadata
        m = self.embed_metadata(
            metadata_tensor,
            score_tensor,
            constraints_location=constraints_loc
        )
        # === LSTM on constraints ===
        output_constraints, activations_constraint = self.output_lstm_constraints(m)
        hidden = init_hidden_lstm(
            num_layers=self.num_layers,
            batch_size=batch_size,
            lstm_hidden_size=self.num_lstm_generation_units
        )
        offset_seq = torch.cat(
            [to_cuda_variable(torch.zeros(batch_size, 1, self.note_embedding_dim)),
             x[:, :sequence_length - 1, :]
             ], 1)

        # todo dropout only on offset_seq?
        offset_seq = self.drop_input(offset_seq)
        input_seq = torch.cat([offset_seq, output_constraints], 2)
        past_input = input_seq[:, :start_tick, :]
        (_, hidden), activations_gen = lstm_with_activations(
            lstm_list=self.lstm_generation,
            input=past_input,
            hidden=hidden)
        h, c = hidden
        hidden = h[:, 0, :, :], c[:, 0, :, :]

        final_weights = [[] for i in range(num_voices)]
        for tick_index in range(start_tick-1, end_tick-1):
            voice_index = tick_index % num_voices
            time_index = (tick_index - voice_index) // num_voices
            next_voice_index = (tick_index + 1) % num_voices
            next_time_index = (tick_index + 1 - next_voice_index) // num_voices

            if tick_index == -1:
                last_start_symbol = 0  # gen_chorale[-1, 0]
                last_start_symbol = torch.from_numpy(np.array([last_start_symbol]))[None, :]
                time_slice = self.note_embeddings[-1](
                    to_cuda_variable(last_start_symbol)
                )
            else:
                time_slice = gen_chorale[:, voice_index:voice_index+1, time_index]
                # time_slice = torch.from_numpy(np.array([time_slice]))[None, :]
                note = self.note_embeddings[voice_index](
                    to_cuda_variable(time_slice)
                )
                time_slice = note

            time_slice_cat = torch.cat((time_slice, output_constraints[:, tick_index+1:tick_index+2, :]), 2)

            (output_gen, hidden), activations_generation = lstm_with_activations(
                lstm_list=self.lstm_generation,
                input=time_slice_cat, hidden=hidden)
            h, c = hidden
            hidden = h[:, 0, :, :], c[:, 0, :, :]

            weights = F.relu(self.linear_1(output_gen[:, 0, :]), inplace=True)
            weights = self.linear_ouput_notes[next_voice_index](weights)

            # compute predictions
            # temperature
            weights = weights
            preds = F.softmax(weights)
            final_weights[voice_index].append(weights.unsqueeze(1))

            # first batch element
            preds = to_numpy(preds[0])
            new_pitch_index = np.argmax(preds)

            gen_chorale[:, next_voice_index, next_time_index] = int(new_pitch_index)
        for i in range(num_voices):
            final_weights[i] = torch.cat(final_weights[i], 1)
        return final_weights, gen_chorale

    def _forward_tf(self, score_tensor: Variable, metadata_tensor: Variable, constraints_loc):
        batch_size, num_voices, chorale_length = score_tensor.size()
        sequence_length = num_voices * chorale_length

        # === embed as wrapped sequence ===
        # --- chorale
        x = self.embed_tensor_score(score_tensor)

        # --- metadata
        m = self.embed_metadata(
            metadata_tensor,
            score_tensor,
            constraints_location=constraints_loc
        )

        # === LSTM on constraints ===
        output_constraints, activations_constraint = self.output_lstm_constraints(m)

        # === LSTM on notes ===
        offset_seq = torch.cat(
            [to_cuda_variable(torch.zeros(batch_size, 1, self.note_embedding_dim)),
             x[:, :sequence_length - 1, :]
             ], 1)

        # todo dropout only on offset_seq?
        offset_seq = self.drop_input(offset_seq)

        input = torch.cat([offset_seq, output_constraints], 2)
        # todo remove dropout?
        # input = self.dropout_input(input)
        hidden = init_hidden_lstm(num_layers=self.num_layers,
                                  batch_size=batch_size,
                                  lstm_hidden_size=self.num_lstm_generation_units)
        # get reg
        (output_gen, hidden), activations_gen = lstm_with_activations(
            lstm_list=self.lstm_generation,
            input=input,
            hidden=hidden)

        # distributed NN on output
        weights = [F.relu(self.linear_1(time_slice), inplace=True)
                   for time_slice
                   in output_gen.split(split_size=1,
                                       dim=1)]
        weights = torch.cat(weights, 1)
        weights = weights.view(batch_size, chorale_length, num_voices, self.num_units_linear)

        # CrossEntropy includes a LogSoftMax layer
        weights = [
            linear_layer(voice[:, :, 0, :])
            for voice, linear_layer
            in zip(weights.split(split_size=1, dim=2), self.linear_ouput_notes)
        ]
        lstm_activations = [
            activations_gen, activations_constraint
        ]
        return weights, lstm_activations

    def forward(
            self,
            score_tensor: Variable,
            metadata_tensor: Variable,
            constraints_loc,
            start_tick=None,
            end_tick=None,
            train=True
    ):
        # todo binary mask?
        """
        :param score_tensor: (batch, num_voices, length in ticks)
        :param metadata_tensor: (batch, num_voices, length in ticks, num_metadatas)
        :param constraints_loc: torch Tensor, with the constraint locations
        :param start_tick: int,
        :param end_tick
        :param train: bool, specifies if being used for training
        :return: list of probabilities per voice (batch, chorale_length, num_notes)
        """
        if self.use_teacher_forcing and train:
            teacher_forcing = random.random() <= self.teacher_forcing_prob
        else:
            teacher_forcing = False

        if teacher_forcing:
            weights, add_args = self._forward_tf(score_tensor, metadata_tensor, constraints_loc)
        else:
            weights, add_args = self._forward_no_tf(score_tensor, metadata_tensor, constraints_loc)
        weights = [weight_per_voice[:, (constraints_loc[0, i, :] == 0).nonzero().squeeze(), :] for i, weight_per_voice in enumerate(weights)]
        return weights, add_args

    def drop_input(self, x):
        """
        :param x: (batch_size, seq_length, num_features)
        :return:
        """
        return self.dropout_layer(x[:, :, :, None])[:, :, :, 0]

    def embed_tensor_score(self, tensor_score):
        separate_voices = tensor_score.split(split_size=1, dim=1)
        separate_voices = [
            embedding(voice[:, 0, :])[:, None, :, :]
            for voice, embedding
            in zip(separate_voices, self.note_embeddings)
        ]
        x = torch.cat(separate_voices, 1)
        x = self.flatten_tensor_score(chorale=x)
        return x

    def output_lstm_constraints(self, flat_embedded_metadata):
        """

        :param flat_embedded_metadata: (batch_size, length, total_embedding_dim)
        :return:
        """
        batch_size = flat_embedded_metadata.size(0)
        hidden = init_hidden_lstm(num_layers=self.num_layers,
                             batch_size=batch_size,
                             lstm_hidden_size=self.num_lstm_constraints_units
                             )
        # reverse seq
        idx = [i for i in range(flat_embedded_metadata.size(1) - 1, -1, -1)]
        idx = to_cuda_variable(torch.LongTensor(idx))
        flat_embedded_metadata = flat_embedded_metadata.index_select(1, idx)
        (output_constraints, hidden), activations_constraint = lstm_with_activations(
            self.lstm_constraint,
            flat_embedded_metadata,
            hidden)
        output_constraints = output_constraints.index_select(1, idx)
        return output_constraints, activations_constraint

    def embed_metadata(self, metadata, tensor_score=None, constraints_location=None):
        """

        :param metadata: (batch_size, num_voices, chorale_length, num_metadatas)
        :param tensor_score:
        :param constraints_location
        :return: (batch_size, num_voices * chorale_length, embedding_dim * num_metadatas
        + note_embedding_dim * (1 if chorale else 0))
        """
        # todo problems when adding unary constraints
        batch_size, num_voices, chorale_length, _ = metadata.size()
        m = self.flatten_metadata(metadata=metadata)
        separate_metadatas = m.split(split_size=1,
                                     dim=2)
        separate_metadatas = [
            embedding(separate_metadata[:, :, 0])[:, :, None, :]
            for separate_metadata, embedding
            in zip(separate_metadatas, self.metadata_embeddings)
        ]

        # todo how to merge? multiply or concat
        m = torch.cat(separate_metadatas, 2)
        # concat all
        m = m.view((batch_size, num_voices * chorale_length, -1))

        # append unary constraints
        if tensor_score is not None:
            masked_tensor_score = self.mask_tensor_score(tensor_score,
                                                         constraints_location=constraints_location)
            masked_tensor_embed = self.embed_tensor_score(masked_tensor_score)
            m = torch.cat([m, masked_tensor_embed], 2)
        return m

    def mask_tensor_score(self, tensor_score, constraints_location=None):
        """
        (batch_size, num_voices, chorale_length)
        :param tensor_score:
        :param constraints_location
        :return:
        """
        p = random.random() * 0.5
        if constraints_location is None:
            constraints_location = to_cuda_variable((torch.rand(*tensor_score.size()) < p).long())
        else:
            assert constraints_location.size() == tensor_score.size()
            constraints_location = to_cuda_variable(constraints_location)

        batch_size, num_voices, chorale_length = tensor_score.size()
        no_constraint = torch.from_numpy(
            np.array([len(note2index)
                      for note2index in self.dataset.note2index_dicts])
        )
        no_constraint = no_constraint[None, :, None]
        no_constraint = no_constraint.long().clone().repeat(batch_size, 1, chorale_length)
        no_constraint = to_cuda_variable(no_constraint)
        return tensor_score * constraints_location + no_constraint * (1 - constraints_location)

    def generation(self,
                   tensor_score,
                   tensor_metadata,
                   temperature,
                   time_index_range):
        # process arguments
        if tensor_score is None:
            score_gen = self.dataset.iterator_gen()
            original_score = next(score_gen)
            original_score = next(score_gen)
            original_score = next(score_gen)
        else:
            original_score = self.dataset.tensor_to_score(tensor_score)

        trans_interval = self.dataset.get_transpostion_interval_from_semitone(0)
        (tensor_score,
         tensor_metadata) = self.dataset.transposed_score_and_metadata_tensors(
            original_score,
            trans_interval
        )
        print('Size of Original Chorale: ', tensor_score.size())
        constraints_location = torch.zeros_like(tensor_score)
        a, b = time_index_range
        if a > 0:
            constraints_location[:, :a] = 1
        if b < constraints_location.size(1) - 1:
            constraints_location[:, b:] = 1
        # print(constraints_location)
        score, gen_chorale, tensor_metadata = self.generate(
            tensor_score=tensor_score,
            tensor_metadata=tensor_metadata,
            constraints_location=constraints_location,
            temperature=temperature)
        original_score.show()
        return score, gen_chorale, tensor_metadata

    def generate(self,
                 tensor_score,
                 tensor_metadata,
                 constraints_location,
                 temperature=1.):
        self.eval()
        tensor_score = to_cuda_variable(tensor_score)

        num_voices, chorale_length, num_metadatas = tensor_metadata.size()

        # generated chorale
        gen_chorale = self.dataset.empty_score_tensor(chorale_length)

        m = to_cuda_variable(tensor_metadata[None, :, :, :])
        m = self.embed_metadata(m, tensor_score[None, :, :],
                                constraints_location=constraints_location[None, :, :])

        output_constraints, activations_constraints = self.output_lstm_constraints(m)

        hidden = init_hidden_lstm(
            num_layers=self.num_layers,
            batch_size=1,
            lstm_hidden_size=self.num_lstm_generation_units
        )

        print(tensor_score)
        # 1 bar of start symbols
        # todo check
        for tick_index in range(self.dataset.num_voices
                                * 4
                                * self.dataset.subdivision - 1):
            voice_index = tick_index % self.dataset.num_voices
            # notes
            time_slice = gen_chorale[voice_index, 0]
            time_slice = torch.from_numpy(np.array([time_slice]))[None, :]
            note = self.note_embeddings[voice_index](
                to_cuda_variable(time_slice)
            )
            time_slice = note
            # concat with first metadata
            # todo wrong
            time_slice_cat = torch.cat(
                (time_slice, output_constraints[:,
                             tick_index + 1: tick_index + 2, :]
                 ), 2)

            (output_gen, hidden), activations_generation = lstm_with_activations(
                lstm_list=self.lstm_generation,
                input=time_slice_cat,
                hidden=hidden
            )
            h, c = hidden
            hidden = h[:, 0, :, :], c[:, 0, :, :]
        # generation:
        for tick_index in range(-1, chorale_length * num_voices - 1):
            voice_index = tick_index % num_voices
            time_index = (tick_index - voice_index) // num_voices
            next_voice_index = (tick_index + 1) % num_voices
            next_time_index = (tick_index + 1 - next_voice_index) // num_voices

            if tick_index == -1:
                last_start_symbol = 0  # gen_chorale[-1, 0]
                last_start_symbol = torch.from_numpy(np.array([last_start_symbol]))[None, :]
                time_slice = self.note_embeddings[-1](
                    to_cuda_variable(last_start_symbol)
                )
            else:
                time_slice = gen_chorale[voice_index, time_index]
                time_slice = torch.from_numpy(np.array([time_slice]))[None, :]
                note = self.note_embeddings[voice_index](
                    to_cuda_variable(time_slice)
                )
                time_slice = note

            time_slice_cat = torch.cat(
                (time_slice, output_constraints[:,
                             tick_index + 1: tick_index + 2, :]
                 ), 2
            )

            (output_gen, hidden), activations_generation = lstm_with_activations(
                lstm_list=self.lstm_generation,
                input=time_slice_cat, hidden=hidden)
            h, c = hidden
            hidden = h[:, 0, :, :], c[:, 0, :, :]

            weights = F.relu(self.linear_1(output_gen[:, 0, :]), inplace=True)
            weights = self.linear_ouput_notes[next_voice_index](weights)

            # compute predictions
            # temperature
            weights = weights * temperature
            preds = F.softmax(weights)

            # first batch element
            preds = to_numpy(preds[0])
            new_pitch_index = np.random.choice(np.arange(
                self.num_notes_per_voice[next_voice_index]
            ), p=preds)
            # new_pitch_index = np.argmax(preds)

            gen_chorale[next_voice_index, next_time_index] = int(new_pitch_index)

        # # show original chorale
        #score_original = self.dataset.tensor_to_score(
        #    tensor_score.cpu())
        #score_original.show()
        print(gen_chorale)
        score = self.dataset.tensor_to_score(gen_chorale)
        return score, gen_chorale, tensor_metadata


class AnticipationRNNBaseline(ConstraintModelGaussianReg):
    def __init__(self, dataset: MusicDataset,
                 note_embedding_dim=20,
                 metadata_embedding_dim=30,
                 num_lstm_constraints_units=256,
                 num_lstm_generation_units=256,
                 linear_hidden_size=128,
                 num_layers=1,
                 dropout_input_prob=0.2,
                 dropout_prob=0.5,
                 unary_constraint=False,
                 teacher_forcing=True
                 ):
        super(AnticipationRNNBaseline, self).__init__(
            dataset=dataset,
            note_embedding_dim=note_embedding_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            num_lstm_constraints_units=num_lstm_constraints_units,
            num_lstm_generation_units=num_lstm_generation_units,
            linear_hidden_size=linear_hidden_size,
            num_layers=num_layers,
            dropout_input_prob=dropout_input_prob,
            dropout_prob=dropout_prob,
            unary_constraint=unary_constraint,
            teacher_forcing=teacher_forcing
        )

    def __repr__(self):
        filestr = f'AnticipationRNNBaseline(' \
            f'{self.dataset.__repr__()},' \
            f'{self.note_embedding_dim},' \
            f'{self.metadata_embedding_dim},' \
            f'{self.num_lstm_constraints_units},' \
            f'{self.num_lstm_generation_units},' \
            f'{self.num_units_linear},' \
            f'{self.num_layers},' \
            f'{self.dropout_input_prob},' \
            f'{self.dropout_prob},' \
            f'{self.unary_constraint},' \
            f')'
        if self.use_teacher_forcing:
            filestr += ',tf'
        else:
            filestr += ',no_tf'
        return filestr