import os
from torch import nn
import random

from DatasetManager.the_session.folk_dataset import FolkDatasetNBars
from utils.helpers import *
from utils.model import *
from MeasureVAE.measure_vae import MeasureVAE


class LatentRNNAblations(Model):
    def __init__(self,
                 dataset,
                 vae_model: MeasureVAE,
                 num_rnn_layers,
                 rnn_hidden_size,
                 dropout,
                 rnn_class,
                 auto_reg=False,
                 teacher_forcing=True,
                 type='past'):
        """
        Initializes a LatentRNN class object
        :param dataset: FolkDatasetNBars object
        :param vae_model: MeasureVAE object, pre-trained
        :param num_rnn_layers: int,
        :param rnn_hidden_size: int,
        :param dropout: float, from 0. to 1.,
        :param rnn_class: torch.nn.RNN identifier
        :param auto_reg: bool, model is auto-regressive if TRUE
        :param type: string, "past" or "future"
        """
        super(LatentRNNAblations, self).__init__()

        # initialize members
        self.dataset = dataset.__repr__()
        self.vae_model = vae_model
        self.auto_reg = auto_reg
        if self.auto_reg:
            self.use_teacher_forcing = teacher_forcing
        else:
            self.use_teacher_forcing = False
        self.teacher_forcing_prob = 0.5
        for param in self.vae_model.parameters():
            param.requires_grad = False
        print('Freeze the ', self.vae_model.__repr__(), ' model.')
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout
        self.z_dim = self.vae_model.latent_space_dim
        self.rnn_class = rnn_class
        self.bidirectional = True
        self.rnn_num_direction = 2 if self.bidirectional else 1
        self.type = type
        # define network layers
        self.context_rnn_past = self.rnn_class(
            input_size=self.z_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_rnn_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        self.context_rnn_future = self.rnn_class(
            input_size=self.z_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_rnn_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        if self.auto_reg:
            self.gen_rnn_input_dim = self.z_dim
        else:
            self.gen_rnn_input_dim = 1
            self.x_0 = nn.Parameter(data=torch.randn(1, 1, self.gen_rnn_input_dim))
        self.generation_rnn = self.rnn_class(
            input_size=self.gen_rnn_input_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_rnn_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        self.generation_linear = nn.Linear(self.rnn_hidden_size * self.rnn_num_direction, self.z_dim)
        self.xavier_initialization()

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.filepath = os.path.join(cur_dir, 'models/',
                                     self.__repr__())

    def __repr__(self):
        """
        String Representation of class
        :return: str,
        """
        filestr = f'LatentRNN(' \
                  f'{self.type}' \
                  f'{self.dataset}' \
                  f'{self.rnn_class},' \
                  f'{self.num_rnn_layers},' \
                  f'{self.rnn_hidden_size},' \
                  f'{self.dropout},' \
                  f')'
        if self.auto_reg:
            filestr += 'auto_reg'
        if self.use_teacher_forcing:
            filestr += ',tf'
        else:
            filestr += ',no_tf'
        return filestr

    def forward(self, past_context, future_context, target, measures_to_generate, train=True):
        """
        Computes the forward pass for the LatentRNN model, overrides the torch method
        :param past_context: torch Variable,
                (batch_size, num_measures_past, measure_seq_len)
        :param future_context: torch Variable,
                (batch_size, num_measures_future, measure_seq_len)
        :param target: torch Variable,
                (batch_size, num_measures, target, measure_seq_len)
        :param measures_to_generate: int, number of measures to generate
        :param train: bool,
        :return: weights: torch Variable: softmax weights for the predicted measure score
                (batch_size, measures_to_generate * measure_seq_len, num_notes)
                samples: torch Variable: indices of the predicted measure score
                (batch_size, measures_to_generate, measure_seq_len)
                gen_z: torch Variable, predicted latent space vectors,
                (batch_size, measures_to_generate, self.z_dim)
        """
        batch_size, _, measure_seq_len = past_context.size()

        # get sequence of latent space vectors via VAE forward pass
        zp = self.get_z_seq(past_context)
        zf = self.get_z_seq(future_context)
        zt = self.get_z_seq(target)

        # get context vectors via context_rnn forward pass
        context_past = self.forward_context(zp, type="past")
        context_future = self.forward_context(zf, type="future")

        # concatenate contexts
        if self.type == "past":
            comb_context = context_past
        else:
            comb_context = context_future

        if self.use_teacher_forcing and train:
            teacher_forcing = random.random() < self.teacher_forcing_prob
        else:
            teacher_forcing = False

        # pass through generation lstm
        if teacher_forcing:
            seed = torch.cat((zp[:, -1, :].unsqueeze(1), zt[:, :-1, :]), 1)
        else:
            seed = zp[:, -1, :].unsqueeze(1)
        weights, samples, gen_z = self.forward_generation(
            comb_context,
            measures_to_generate,
            seed,
            measure_seq_len,
            teacher_forcing
        )
        return weights, samples, gen_z

    def get_z_seq(self, measures_tensor):
        """
        Computes the forward pass through the vae encoder to return a seq of latent vectors
        :param measures_tensor: torch Variable,
                (batch_size, num_measures, measure_seq_len)
        :return: torch Variable,
                (batch_size, num_measures, self.z_dim)
        """
        batch_size, num_measures, measure_seq_len = measures_tensor.size()
        measures_tensor = measures_tensor.view(-1, measure_seq_len)
        z_dists = self.vae_model.encoder(measures_tensor)
        z = z_dists.rsample()
        z = z.view(batch_size, -1, self.z_dim)
        return z

    def forward_context(self, z, type):
        """
        Computes the context vector for past or future measures
        :param z: torch Variable,
                (batch_size, num_past_measures, self.z_dim)
        :param type: str, "past" or "future"
        :return: torch Variable
                (self.num_rnn_layers * self.num_directions, batch_size, self.rnn_hidden_size)
        """
        batch_size = z.size(0)
        hidden = self.hidden_init(batch_size)
        if type == "past":
            _, hidden = self.context_rnn_past(z, hidden)
        elif type == "future":
            _, hidden = self.context_rnn_future(z, hidden)
        else:
            raise ValueError
        return hidden

    def hidden_init(self, batch_size):
        """
        Initializes the hidden state for the context RNNs
        :param batch_size: int,
        :return: torch tensor,
                (self.num_rnn_layers * self.rnn_num_directions, batch_size, self.rnn_hidden_size)
        """
        h = to_cuda_variable(
            torch.zeros(
                self.num_rnn_layers * self.rnn_num_direction,
                batch_size,
                self.rnn_hidden_size
            )
        )
        return h

    def forward_generation(self, context_vector, measures_to_gen, seed, measure_seq_len, teacher_forcing=False):
        """
        Computes the forward pass for the generation RNN and linear layers
        :param context_vector: torch Variable
                (self.num_rnn_layers * self.num_directions, batch_size, self.rnn_hidden_size)
        :param measures_to_gen: int, number of measures to generate
        :param seed, torch Variable, seed for generation, uses ground truth if teacher_forcing is TRUE
        :param measure_seq_len: int, num of ticks in a measure
        :param teacher_forcing: bool, use teacher forcing if TRUE
        :return: z_out: torch Variable,
                (batch_size, measures_to_gen, self.z_dim)
        """
        # pass through RNN
        batch_size = context_vector.size(1)
        if self.auto_reg:
            gen_rnn_input = seed
        else:
            gen_rnn_input = self.x_0.expand(batch_size, measures_to_gen, -1)

        if teacher_forcing or not self.auto_reg:
            gen_rnn_out, _ = self.generation_rnn(gen_rnn_input, context_vector)
            gen_rnn_out = gen_rnn_out.contiguous().view(batch_size * measures_to_gen, -1)
            z_out = self.generation_linear(gen_rnn_out).contiguous().view(batch_size, measures_to_gen, -1)
            weights = []
            samples = []
            dummy_measure_tensor = to_cuda_variable(torch.zeros(batch_size, measure_seq_len))
            for i in range(measures_to_gen):
                w, s = self.vae_model.decoder(z_out[:, i, :], dummy_measure_tensor, train=False)
                samples.append(s)
                weights.append(w.unsqueeze(1))
        else:
            hidden = context_vector
            z_out = []
            weights = []
            samples = []
            dummy_measure_tensor = to_cuda_variable(torch.zeros(batch_size, measure_seq_len))
            for i in range(measures_to_gen):
                # compute generation_rnn output
                rnn_out, hidden = self.generation_rnn(gen_rnn_input, hidden)
                # compute generation_linear output
                rnn_out = rnn_out.contiguous().view(batch_size, -1)
                gen_z = self.generation_linear(rnn_out).contiguous().view(batch_size, 1, -1)
                z_out.append(gen_z)
                # compute forward pass through VAE decoder
                w, s = self.vae_model.decoder(gen_z.squeeze(1), dummy_measure_tensor, train=False)
                samples.append(s)
                weights.append(w.unsqueeze(1))
                # pass through VAE encoder to get new input for RNN
                gen_rnn_input = self.get_z_seq(s)
            z_out = torch.cat(z_out, 1)
        samples = torch.cat(samples, 2)
        weights = torch.cat(weights, 1)
        return weights, samples, z_out

    def save(self):
        """
        Saves the model
        :return: None
        """
        torch.save(self.state_dict(), self.filepath)
        print(f'Model {self.__repr__()} saved')

    def load(self, cpu=False):
        """
        Loads the model
        :param cpu: bool, specifies if the model should be loaded on the CPU
        :return: None
        """
        if cpu:
            self.load_state_dict(
                torch.load(
                    self.filepath,
                    map_location=lambda storage,
                    loc: storage
                )
            )
        else:
            self.load_state_dict(torch.load(self.filepath))
        print(f'Model {self.__repr__()} loaded')

    def xavier_initialization(self):
        """
        Initializes the network params
        :return: None
        """
        for name, param in self.context_rnn_past.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.context_rnn_future.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.generation_rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.generation_linear.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
