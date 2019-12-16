import torch
from scipy import stats
from random import shuffle

from music21 import interval
from torch.utils.data import TensorDataset

from DatasetManager.music_dataset import MusicDataset
from DatasetManager.exceptions import *
from DatasetManager.the_session.folk_data_helpers import *


class FolkDataset(MusicDataset):
    def __init__(
            self,
            name,
            corpus_it_gen=None,
            metadatas=None,
            sequences_size=32,
            cache_dir=None
    ):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over the files (as music21 scores)
        :param name
        :param sequences_size: in beats
        :param subdivision: number of sixteenth notes per beat
        :param cache_dir: directory where the tensor_dataset is stored
        """
        super(FolkDataset, self).__init__(cache_dir=cache_dir)
        self.name = name
        self.corpus_it_gen = corpus_it_gen
        self.num_melodies = self.corpus_it_gen.num_elements
        self.NOTES = 0
        self.num_voices = 1
        self.pitch_range = [55, 84]
        self.tick_values = tick_values
        self.subdivision = len(self.tick_values)
        self.tick_durations = self.compute_tick_durations()
        self.seq_size_in_beats = sequences_size
        self.metadatas = metadatas
        if self.metadatas:
            for metadata in self.metadatas:
                if metadata.name == 'beatmarker':
                    self.beat_index2symbol_dicts = metadata.beat_index2symbol_dicts
                    self.beat_symbol2index_dicts = metadata.beat_symbol2index_dicts
        self.index2note_dicts = None
        self.note2index_dicts = None
        self.dicts_dir = os.path.join(
            self.cache_dir,
            'dicts',
        )
        if not os.path.exists(self.dicts_dir):
            os.mkdir(self.dicts_dir)
        self.dict_path = os.path.join(
            self.dicts_dir, 'index_dicts.txt'
        )

    def __repr__(self):
        return f'FolkDataset(' \
               f'{self.name},' \
               f'{[metadata.name for metadata in self.metadatas]},' \
               f'{self.seq_size_in_beats},' \
               f'{self.subdivision})' \
               f'{self.num_melodies}'

    def iterator_gen(self):
        return (score
                for score in self.corpus_it_gen()
                )

    def compute_tick_durations(self):
        """
        Computes the tick durations
        """
        diff = [n - p
                for n, p in zip(self.tick_values[1:], self.tick_values[:-1])]
        diff = diff + [1 - self.tick_values[-1]]
        return diff

    def get_score_tensor(self, score):
        """
        Extract the lead tensor from the lead sheet
        :param score: music21 score object
        :return: lead_tensor
        """
        eps = 1e-4
        notes, _ = notes_and_chords(score)
        if not score_on_ticks(score, self.tick_values):
            raise LeadsheetParsingException(
                f'Score {score.metadata.title} has notes not on ticks')

        # add entries to dictionaries if not present
        # should only be called by make_tensor_dataset when transposing
        list_note_strings_and_pitches = [(n.nameWithOctave, n.pitch.midi)
                                         for n in notes
                                         if n.isNote]
        note2index = self.note2index_dicts[self.NOTES]
        index2note = self.index2note_dicts[self.NOTES]
        pitch_range = self.pitch_range
        min_pitch, max_pitch = pitch_range
        for note_name, pitch in list_note_strings_and_pitches:
            # if out of range
            if pitch < min_pitch or pitch > max_pitch:
                note_name = OUT_OF_RANGE
            if note_name not in note2index:
                new_index = len(note2index)
                index2note.update({new_index: note_name})
                note2index.update({note_name: new_index})
                print('Warning: Entry ' + str(
                    {new_index: note_name}) + ' added to dictionaries')
                self.update_index_dicts()

        # construct sequence
        j = 0
        i = 0
        length = int(score.highestTime * self.subdivision)
        t = np.zeros((length, 2))
        is_articulated = True
        num_notes = len(notes)
        current_tick = 0
        while i < length:
            if j < num_notes - 1:
                if notes[j + 1].offset > current_tick + eps:
                    t[i, :] = [note2index[standard_name(notes[j])],
                               is_articulated]
                    i += 1
                    current_tick += self.tick_durations[
                        (i - 1) % len(self.tick_values)]
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [note2index[standard_name(notes[j])],
                           is_articulated]
                i += 1
                is_articulated = False
        lead = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * note2index[SLUR_SYMBOL]
        # convert to torch tensor
        lead_tensor = torch.from_numpy(lead).long()[None, :]
        return lead_tensor  # , chord_tensor

    def get_metadata_tensor(self, score):
        """
        Extract the metadata tensor (beat markers) from the lead sheet
        :param score: music21 score object
        :return: metadata_tensor
        """
        md = []
        if self.metadatas:
            for metadata in self.metadatas:
                sequence_metadata = torch.from_numpy(
                    metadata.evaluate(score, self.subdivision)).long().clone()
                square_metadata = sequence_metadata.repeat(self.num_voices, 1)
                md.append(
                    square_metadata[:, :, None]
                )
        # compute length
        lead_length = int(score.highestTime * self.subdivision)
        # add voice indexes
        voice_id_metada = torch.from_numpy(np.arange(self.num_voices)).long().clone()
        square_metadata = torch.transpose(
            voice_id_metada.repeat(lead_length, 1),
            0,
            1
        )
        md.append(square_metadata[:, :, None])

        all_metadata = torch.cat(md, 2)
        return all_metadata
        # all_metadata = torch.cat(md, 2)
        # return all_metadata

    @staticmethod
    def get_transpostion_interval_from_semitone(semi_tone):
        """
        Converts semi-tone to music21 interval
        :param semi_tone: int, -12 to +12
        :return: music21.Interval object
        """
        # compute the most "natural" interval given a number of semi-tones
        interval_type, interval_nature = interval.convertSemitoneToSpecifierGeneric(
            semi_tone)
        transposition_interval = interval.Interval(
            str(interval_nature) + interval_type)
        return transposition_interval

    def transposed_score_and_metadata_tensors(self, score, trans_int):
        """
        Convert chorale to a couple (chorale_tensor, metadata_tensor),
        the original chorale is transposed semi_tone number of semi-tones

        :param score: music21 object
        :param trans_int: music21.interval.Interval object
        :return: couple of tensors
        """
        # TODO: implement this properly. 
        try:
            score_transposed = score.transpose(trans_int)
        except ValueError as e:
            raise LeadsheetParsingException(f'Leadsheet {score.metadata.title} '
                                            f'not properly formatted')
        transposed_score_tensor = self.get_score_tensor(score_transposed)
        transposed_metadata_tensor = self.get_metadata_tensor(score_transposed)
        return transposed_score_tensor, transposed_metadata_tensor

    def make_tensor_dataset(self):
        self.compute_index_dicts()
        print('Making tensor dataset')
        lead_tensor_dataset = []
        metadata_tensor_dataset = []
        count = 0
        for score_id, score in tqdm(enumerate(self.corpus_it_gen())):
            if not self.is_in_range(score):
                continue
            try:
                if count > self.num_melodies:
                    break
                count += 1
                lead_tensor = self.get_score_tensor(score)
                metadata_tensor = self.get_metadata_tensor(score)
                # main loop - lead
                for offset_start in range(
                        -self.seq_size_in_beats + 1,
                        int(score.highestTime)
                ):
                    offset_end = offset_start + self.seq_size_in_beats
                    local_lead_tensor = self.extract_score_tensor_with_padding(
                        tensor=lead_tensor,
                        start_tick=offset_start * self.subdivision,
                        end_tick=offset_end * self.subdivision
                    )
                    local_metadata_tensor = self.extract_metadata_with_padding(
                        tensor_metadata=metadata_tensor,
                        start_tick=offset_start * self.subdivision,
                        end_tick=offset_end * self.subdivision
                    )
                    # append and add batch dimension
                    # cast to int
                    lead_tensor_dataset.append(
                        local_lead_tensor.int()
                    )
                    metadata_tensor_dataset.append(
                        local_metadata_tensor.int()
                    )
            except LeadsheetParsingException as e:
                print(e)
                print(f'For score: {score_id}')
        lead_tensor_dataset = torch.cat(lead_tensor_dataset, 0)
        num_datapoints = lead_tensor_dataset.size()[0]
        lead_tensor_dataset = lead_tensor_dataset.view(
            num_datapoints, 1, -1
        )
        metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)
        num_datapoints, length, num_metadata = metadata_tensor_dataset.size()
        metadata_tensor_dataset = metadata_tensor_dataset.view(
            num_datapoints, 1, length, num_metadata
        )
        dataset = TensorDataset(lead_tensor_dataset, metadata_tensor_dataset)
        print(f'Sizes: {lead_tensor_dataset.size()}')
        print(f'Sizes: {metadata_tensor_dataset.size()}')
        return dataset

    def make_tensor_dataset_full_melody(self):
        """
        Creates tensor and metadata datasets with full length melodies
        Uses a packed padded sequence approach
        """
        pass

    def create_packed_pad_dataset(self, tensor_dataset, longest_seq_len):
        """
        Takes a list of variable length tensors and creates a dataset
        Uses pytorch packed padded sequence to zero-pad appropriately

        :param tensor_dataset: list of tensors differening in dim 1
        :param longest_seq_len: length of the longest sequence
        """
        # get the tensor dimensions
        # interate and pad zeros, keep track of number of zeros padded
        pass

    def transposed_score_tensor(self, score, semi_tone):
        """
        Convert score to a tensor,
        the original lead is transposed semi_tone number of semi-tones

        :param score: music21 object
        :param semi_tone: int, number of semi-tones to transpose by
        :return: transposed lead tensor
        """
        # transpose
        # compute the most "natural" interval given a number of semi-tones
        transposition_interval = self.get_transpostion_interval_from_semitone(semi_tone)
        score_tranposed = score.transpose(transposition_interval)
        if not self.is_in_range(score_tranposed):
            return None
        lead_tensor = self.get_score_tensor(score_tranposed)
        return lead_tensor

    def extract_score_tensor_with_padding(self, 
                                          tensor,
                                          start_tick,
                                          end_tick):
        """

        :param tensor: (batch_size, length)
        :param start_tick:
        :param end_tick:
        :return: (batch_size, end_tick - start_tick)
        """
        assert start_tick < end_tick
        # assert end_tick > 0
        batch_size, length = tensor.size()
        symbol2index = self.note2index_dicts[self.NOTES]
        padded_tensor = []
        if start_tick < 0:
            start_symbols = np.array([symbol2index[START_SYMBOL]])
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            start_symbols = start_symbols.repeat(batch_size, -start_tick)
            # start_symbols[-1] = symbol2index[START_SYMBOL]
            padded_tensor.append(start_symbols)

        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length

        padded_tensor.append(tensor[:, slice_start: slice_end])

        if end_tick > length:
            end_symbols = np.array([symbol2index[END_SYMBOL]])
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            end_symbols = end_symbols.repeat(batch_size, end_tick - length)
            # end_symbols[0] = symbol2index[END_SYMBOL]
            padded_tensor.append(end_symbols)

        padded_tensor = torch.cat(padded_tensor, 1)
        return padded_tensor

    def extract_metadata_with_padding(self, tensor_metadata,
                                      start_tick, end_tick):
        """

        :param tensor_metadata: (num_voices, length, num_metadatas)
        last metadata is the voice_index
        :param start_tick:
        :param end_tick:
        :return:
        """
        assert start_tick < end_tick
        # assert end_tick > 0
        num_voices, length, num_metadatas = tensor_metadata.size()
        padded_tensor_metadata = []

        if start_tick < 0:
            # TODO fix PAD symbol in the beginning and end
            start_symbols = np.zeros((self.num_voices, -start_tick, num_metadatas))
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            padded_tensor_metadata.append(start_symbols)

        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length
        padded_tensor_metadata.append(tensor_metadata[:, slice_start: slice_end, :])

        if end_tick > length:
            end_symbols = np.zeros((self.num_voices, end_tick - length, num_metadatas))
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            padded_tensor_metadata.append(end_symbols)

        padded_tensor_metadata = torch.cat(padded_tensor_metadata, 1)
        return padded_tensor_metadata

    def compute_index_dicts(self):
        if os.path.exists(self.dict_path):
            print('Dictionaries already exists. Reading them now')
            f = open(self.dict_path, 'r')
            dicts = [line.rstrip('\n') for line in f]
            assert (len(dicts) == 2)  # must have 2 dictionaries
            self.index2note_dicts = eval(dicts[0])
            self.note2index_dicts = eval(dicts[1])
            return

        # self.compute_beatmarker_dicts()
        print('Computing note index dicts')
        self.index2note_dicts = [
            {} for _ in range(self.num_voices)
        ]
        self.note2index_dicts = [
            {} for _ in range(self.num_voices)
        ]

        # create and add additional symbols
        note_sets = [set() for _ in range(self.num_voices)]
        for note_set in note_sets:
            note_set.add(SLUR_SYMBOL)
            note_set.add(START_SYMBOL)
            note_set.add(END_SYMBOL)
            # note_set.add(PAD_SYMBOL)

        # get all notes
        # iteratre through all scores and fill in the notes 
        # for tune_filepath in tqdm(self.valid_tune_filepaths):
        count = 0
        for _, score in tqdm(enumerate(self.corpus_it_gen())):
            # score = self.get_score_from_path(tune_filepath)
            # part is either lead or chords as lists
            if count > self.num_melodies:
                break
            count += 1
            for part_id, part in enumerate(notes_and_chords(score)):
                for n in part:
                    note_sets[part_id].add(standard_name(n))

        # create tables
        for note_set, index2note, note2index in zip(note_sets,
                                                    self.index2note_dicts,
                                                    self.note2index_dicts):
            for note_index, note in enumerate(note_set):
                index2note.update({note_index: note})
                note2index.update({note: note_index})

        # write as text file for use later
        self.update_index_dicts()

    def update_index_dicts(self):
        f = open(self.dict_path, 'w')
        f.write("%s\n" % self.index2note_dicts)
        f.write("%s\n" % self.note2index_dicts)
        f.close()

    def is_in_range(self, score):
        """
        Checks if the pitches are within the min and max range

        :param score: music21 score object
        :return: boolean 
        """
        notes, _ = notes_and_chords(score)
        pitches = [n.pitch.midi for n in notes if n.isNote]
        if len(pitches) == 0:
            return False
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        return (min_pitch >= self.pitch_range[0]
                and max_pitch <= self.pitch_range[1])

    def empty_score_tensor(self, score_length):
        """
        
        :param score_length: int, length of the score in ticks
        :return: torch long tensor, initialized with start indices 
        """
        start_symbols = np.array([note2index[START_SYMBOL]
                                  for note2index in self.note2index_dicts])
        start_symbols = torch.from_numpy(start_symbols).long().clone()
        start_symbols = start_symbols.repeat(score_length, 1).transpose(0, 1)
        return start_symbols

    def random_score_tensor(self, score_length):
        """
        
        :param score_length: int, length of the score in ticks
        :return: torch long tensor, initialized with start indices 
        """
        tensor_score = np.array(
            [np.random.randint(len(note2index),
                               size=score_length)
             for note2index in self.note2index_dicts])
        tensor_score = torch.from_numpy(tensor_score).long().clone()
        return tensor_score

    def tensor_to_score(self, tensor_score):
        """
        Converts lead given as tensor_lead to a true music21 score
        :param tensor_score:
        :return:
        """
        slur_index = self.note2index_dicts[self.NOTES][SLUR_SYMBOL]

        score = music21.stream.Score()
        part = music21.stream.Part()
        # LEAD
        dur = 0
        f = music21.note.Rest()
        tensor_lead_np = tensor_score.numpy().flatten()
        for tick_index, note_index in enumerate(tensor_lead_np):
            # if it is a played note
            if not note_index == slur_index:
                # add previous note
                if dur > 0:
                    f.duration = music21.duration.Duration(dur)
                    part.append(f)

                dur = self.tick_durations[tick_index % self.subdivision]
                f = standard_note(self.index2note_dicts[self.NOTES][note_index])
            else:
                dur += self.tick_durations[tick_index % self.subdivision]
        # add last note
        f.duration = music21.duration.Duration(dur)
        part.append(f)
        score.insert(part)
        return score

    def all_transposition_intervals(self, score):
        """
        Finds all the possible transposition intervals from the score
        :param score: music21 score object
        :return: list of music21 interval objects
        """
        min_pitch, max_pitch = score_range(score)
        min_pitch_corpus, max_pitch_corpus = self.pitch_range

        min_transposition = min_pitch_corpus - min_pitch
        max_transposition = max_pitch_corpus - max_pitch

        transpositions = []
        for semi_tone in range(min_transposition, max_transposition + 1):
            interval_type, interval_nature = music21.interval.convertSemitoneToSpecifierGeneric(
                semi_tone)
            transposition_interval = music21.interval.Interval(
                str(interval_nature) + interval_type)
            transpositions.append(transposition_interval)
        return transpositions


class FolkMeasuresDataset(FolkDataset):
    def __repr__(self):
        return f'FolkMeasuresDataset(' \
               f'{self.name},' \
               f'{[metadata.name for metadata in self.metadatas]},' \
               f'{self.subdivision})' \
               f'{self.num_melodies}'

    def make_tensor_dataset(self):
        """

        :return: TensorDataset
        """
        self.compute_index_dicts()
        print('Making measure tensor dataset')
        measure_tensor_dataset = []
        metadata_tensor_dataset = []
        for score_id, score in tqdm(enumerate(self.corpus_it_gen())):
            if not self.is_in_range(score):
                continue
            score_tensor = self.get_score_tensor(score)
            metadata_tensor = self.get_metadata_tensor(score)
            local_measure_tensor = \
                self.split_score_tensor_to_measures(score_tensor)
            local_metadata_tensor = \
                self.split_metadata_tensor_to_measures(metadata_tensor)
            measure_tensor_dataset.append(local_measure_tensor.int())
            metadata_tensor_dataset.append(local_metadata_tensor.int())
        measure_tensor_dataset = torch.cat(measure_tensor_dataset, 0)
        metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)
        dataset = TensorDataset(
            measure_tensor_dataset,
            metadata_tensor_dataset
        )
        print(f'Sizes: {measure_tensor_dataset.size()}')
        print(f'Sizes: {metadata_tensor_dataset.size()}')
        return dataset

    def split_score_tensor_to_measures(self, tensor_score):
        """
        Splits the score tensor to measures

        :param tensor_score: torch tensor, (1, length)
        :return: torch tensor, (num_measures, measure_seq_length)
        """
        batch_size, seq_length = tensor_score.size()
        assert(batch_size == 1)

        # TODO: only works for 4by4 time signatures currently
        measure_seq_length = int(self.subdivision * 4)
        # assert(seq_length % self.subdivision == 0)
        num_measures = int(np.floor(seq_length / measure_seq_length))

        # truncate sequence if needed
        tensor_score = tensor_score[:, :num_measures * measure_seq_length]
        measure_tensor_score = tensor_score.view(num_measures, measure_seq_length)
        return measure_tensor_score

    def split_metadata_tensor_to_measures(self, tensor_metadata):
        """
        Splits the metadata tensor to measures

        :param tensor_metadata: torch tensor, (num_voices, length, num_metadatas)
        :return: torch tensor, (num_measures, measures_seq_length, num_metadatas)
        """
        num_voices, seq_length, num_metadatas = tensor_metadata.size()
        assert(num_voices == 1)
        tensor_metadata = tensor_metadata.view(seq_length, num_metadatas)

        # TODO: only works for 4by4 time signatures currently
        measure_seq_length = int(self.subdivision * 4)
        # assert (seq_length % self.subdivision == 0)
        num_measures = int(np.floor(seq_length / measure_seq_length))

        # truncate sequence if needed
        tensor_metadata = tensor_metadata[:num_measures * measure_seq_length, :]
        measure_tensor_metadata = tensor_metadata.view(num_measures,
                                                       measure_seq_length,
                                                       num_metadatas)
        return measure_tensor_metadata

    # Measure attribute extractors
    def get_num_notes_in_measure(self, measure_tensor):
        """
        Returns the number of notes in each measure of the input normalized by the
        length of the length of the measure representation
        :param measure_tensor: torch Variable,
                (batch_size, measure_seq_len)
        :return: torch Variable containing float tensor ,
                (batch_size)
        """
        _, measure_seq_len = measure_tensor.size()
        slur_index = self.note2index_dicts[self.NOTES][SLUR_SYMBOL]
        rest_index = self.note2index_dicts[self.NOTES]['rest']
        slur_count = torch.sum(measure_tensor == slur_index, 1)
        rest_count = torch.sum(measure_tensor == rest_index, 1)
        note_count = measure_seq_len - (slur_count + rest_count)
        return note_count.float() / float(measure_seq_len)

    def get_note_range_of_measure(self, measure_tensor):
        """
        Returns the note range of each measure of the input normalized by the range
        the dataset
        :param measure_tensor: torch Variable,
                (batch_size, measure_seq_len)
        :return: torch Variable containing float tensor ,
                (batch_size)
        """
        batch_size, measure_seq_len = measure_tensor.size()
        midi_min, midi_max = self.pitch_range
        index2note = self.index2note_dicts[self.NOTES]
        slur_index = self.note2index_dicts[self.NOTES][SLUR_SYMBOL]
        rest_index = self.note2index_dicts[self.NOTES]['rest']
        none_index = self.note2index_dicts[self.NOTES][None]
        has_note = False
        nrange = torch.zeros(batch_size)
        if torch.cuda.is_available():
            nrange = torch.autograd.Variable(nrange.cuda())
        else:
            nrange = torch.autograd.Variable(nrange)
        for i in range(batch_size):
            low_midi = midi_max
            high_midi = midi_min
            for j in range(measure_seq_len):
                index = measure_tensor[i][j].item()
                if index not in (slur_index, rest_index, none_index):
                    has_note = True
                    midi_j = music21.pitch.Pitch(index2note[index]).midi
                    if midi_j < low_midi:
                        low_midi = midi_j
                    if midi_j > high_midi:
                        high_midi = midi_j
            if has_note:
                nrange[i] = high_midi - low_midi
        return nrange.float() / (midi_max - midi_min)

    def get_rhythmic_entropy(self, measure_tensor):
        """
        Returns the rhytmic entropy in a measure of music
        :param measure_tensor: torch Variable,
                (batch_size, measure_seq_len)
        :return: torch Variable,
                (batch_size)
        """
        if measure_tensor.is_cuda:
            measure_tensor_np = measure_tensor.cpu().numpy()
        else:
            measure_tensor_np = measure_tensor.numpy()
        slur_index = self.note2index_dicts[self.NOTES][SLUR_SYMBOL]
        measure_tensor_np[measure_tensor_np != slur_index] = 1
        measure_tensor_np[measure_tensor_np == slur_index] = 0
        ent = stats.entropy(np.transpose(measure_tensor_np))
        ent = torch.from_numpy(np.transpose(ent))
        if torch.cuda.is_available():
            ent = torch.autograd.Variable(ent.cuda())
        else:
            ent = torch.autograd.Variable(ent)
        return ent

    def get_beat_strength(self, measure_tensor):
        """
        Returns the normalized beat strength in a measure of music
        :param measure_tensor: torch Variable,
                (batch_size, measure_seq_len)
        :return: torch Variable,
                (batch_size)
        """
        if measure_tensor.is_cuda:
            measure_tensor_np = measure_tensor.cpu().numpy()
        else:
            measure_tensor_np = measure_tensor.numpy()
        slur_index = self.note2index_dicts[self.NOTES][SLUR_SYMBOL]
        measure_tensor_np[measure_tensor_np != slur_index] = 1
        measure_tensor_np[measure_tensor_np == slur_index] = 0
        weights = np.array([1, 0.008, 0.008, 0.15, 0.008, 0.008])
        weights = np.tile(weights, 4)
        prod = weights * measure_tensor_np
        b_str = np.sum(prod, axis=1)
        if torch.cuda.is_available():
            b_str = torch.autograd.Variable(torch.from_numpy(b_str).cuda())
        else:
            b_str = torch.autograd.Variable(torch.from_numpy(b_str))
        return b_str


class FolkMeasuresDatasetTranspose(FolkMeasuresDataset):
    def __repr__(self):
        return f'FolkMeasuresDatasetTranspose(' \
               f'{self.name},' \
               f'{[metadata.name for metadata in self.metadatas]},' \
               f'{self.subdivision})' \
               f'{self.num_melodies}'

    def make_tensor_dataset(self):
        """

        :return: TensorDataset
        """
        self.compute_index_dicts()
        print('Making measure tensor dataset')
        measure_tensor_dataset = []
        metadata_tensor_dataset = []
        for score_id, score in tqdm(enumerate(self.corpus_it_gen())):
            if not self.is_in_range(score):
                continue
            possible_transpositions = self.all_transposition_intervals(score)
            for trans_int in possible_transpositions:
                score_tensor, metadata_tensor = self.transposed_score_and_metadata_tensors(score, trans_int)
                local_measure_tensor = \
                    self.split_score_tensor_to_measures(score_tensor)
                local_metadata_tensor = \
                    self.split_metadata_tensor_to_measures(metadata_tensor)
                measure_tensor_dataset.append(local_measure_tensor.int())
                metadata_tensor_dataset.append(local_metadata_tensor.int())
        measure_tensor_dataset = torch.cat(measure_tensor_dataset, 0)
        metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)
        dataset = TensorDataset(
            measure_tensor_dataset,
            metadata_tensor_dataset
        )
        print(f'Sizes: {measure_tensor_dataset.size()}')
        print(f'Sizes: {metadata_tensor_dataset.size()}')
        return dataset


class FolkDatasetNBars(FolkMeasuresDataset):
    """
    Class to create n-bar sequences of 4by4 music
    """
    def __init__(self,
                 name,
                 corpus_it_gen=None,  # TODO: NOT BEING USED RIGHT NOW
                 metadatas=None,
                 sequences_size=32,
                 subdivision=4,  # TODO: NOT BEING USED RIGHT NOW
                 cache_dir=None,
                 num_bars=16,
                 train=True):
        super(FolkDatasetNBars, self).__init__(
            name=name,
            corpus_it_gen=corpus_it_gen,
            metadatas=metadatas,
            sequences_size=sequences_size,
            cache_dir=cache_dir
        )
        self.corpus_it_gen.get_valid_tune_filepaths()
        self.train = train
        self.n_bars = num_bars
        self.num_beats_per_bar = 4
        self.seq_size_in_beats = self.num_beats_per_bar * self.n_bars

        self.num_files = None
        self.num_partition = None
        self.dataset_filenames = None
        self.num_dataset_files = None
        # create splits for training and testing sets
        shuffle(self.corpus_it_gen.valid_tune_filenames)
        self.valid_filenames = self.corpus_it_gen.valid_tune_filenames[:self.corpus_it_gen.num_elements]
        self.num_files = len(self.valid_filenames)
        self.num_partition = int(0.9 * self.num_files)
        if self.train:
            self.dataset_filenames = self.valid_filenames[:self.num_partition]
            self.dataset_type = 'train'
        else:
            self.dataset_filenames = self.valid_filenames[self.num_partition:]
            self.dataset_type = 'test'
        self.num_dataset_files = len(self.dataset_filenames)

    def __repr__(self):
        return 'FolkDatasetNBars(' + \
               str(self.n_bars) + \
               str([metadata.name for metadata in self.metadatas]) + \
               ')' + \
               str(self.num_melodies) + '_' + \
               self.dataset_type

    def get_tensor_dataset(self, f, score_tensor_dataset, metadata_tensor_dataset):
        """
        :param f: str, filename
        :param score_tensor_dataset: list,
        :param metadata_tensor_dataset: list,
        :return:
        """
        f = os.path.join(self.corpus_it_gen.raw_dataset_dir, f)
        score = self.corpus_it_gen.get_score_from_path(f, fix_and_expand=True)
        if not self.is_in_range(score):
            return
        # score.show()
        # self.tensor_to_score(self.get_score_tensor(score)).show()
        possible_transpositions = self.all_transposition_intervals(score)
        for trans_int in possible_transpositions:
            score_tensor, metadata_tensor = self.transposed_score_and_metadata_tensors(score, trans_int)
            total_beats = int(score.highestTime)
            for offset_start in range(
                    -self.num_beats_per_bar,
                    total_beats,
                    int(self.seq_size_in_beats)  # 50% overlap
            ):
                offset_end = offset_start + self.seq_size_in_beats
                local_score_tensor = self.extract_score_tensor_with_padding(
                    tensor=score_tensor,
                    start_tick=offset_start * self.subdivision,
                    end_tick=offset_end * self.subdivision
                )
                local_metadata_tensor = self.extract_metadata_with_padding(
                    tensor_metadata=metadata_tensor,
                    start_tick=offset_start * self.subdivision,
                    end_tick=offset_end * self.subdivision
                )
                # append and add batch dimension
                # cast to int
                score_tensor_dataset.append(local_score_tensor.int())
                metadata_tensor_dataset.append(local_metadata_tensor.int())

    def make_tensor_dataset(self):
        """

        :return: Tensor dataset
        """
        self.compute_index_dicts()
        print('Making tensor dataset')
        score_tensor_dataset = []
        metadata_tensor_dataset = []
        for _, f in tqdm(enumerate(self.dataset_filenames)):
            self.get_tensor_dataset(f, score_tensor_dataset, metadata_tensor_dataset)
        score_tensor_dataset = torch.cat(score_tensor_dataset, 0)
        num_datapoints = score_tensor_dataset.size()[0]
        score_tensor_dataset = score_tensor_dataset.view(
            num_datapoints, 1, -1
        )
        metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)
        num_datapoints, length, num_metadata = metadata_tensor_dataset.size()
        metadata_tensor_dataset = metadata_tensor_dataset.view(
            num_datapoints, 1, length, num_metadata
        )
        dataset = TensorDataset(score_tensor_dataset, metadata_tensor_dataset)
        print(f'Sizes: {score_tensor_dataset.size()}')
        print(f'Sizes: {metadata_tensor_dataset.size()}')
        return dataset


if __name__ == '__main__':

    from DatasetManager.dataset_manager import DatasetManager
    from DatasetManager.metadata import BeatMarkerMetadata, TickMetadata

    dataset_manager = DatasetManager()
    metadatas = [
        BeatMarkerMetadata(subdivision=6),
        TickMetadata(subdivision=6)
    ]
    folk_dataset_kwargs = {
        'metadatas': metadatas,
        'sequences_size': 32,
        'num_bars': 1,
        'train': True
    }
    folk_dataset: FolkDataset = dataset_manager.get_dataset(
        name='folk_4by4nbars_chords',
        **folk_dataset_kwargs
    )
    (train_dataloader,
     val_dataloader,
     test_dataloader) = folk_dataset.data_loaders(
        batch_size=100,
        split=(0.7, 0.2)
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))

    folk_dataset_kwargs = {
        'metadatas': metadatas,
        'sequences_size': 32,
        'num_bars': 1,
        'train': False
    }
    folk_dataset: FolkDataset = dataset_manager.get_dataset(
        name='folk_4by4nbars_chords',
        **folk_dataset_kwargs
    )
    (train_dataloader,
     val_dataloader,
     test_dataloader) = folk_dataset.data_loaders(
        batch_size=100,
        split=(0.7, 0.2)
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))

    '''
    for sample_id, (score, _) in tqdm(enumerate(train_dataloader)):
        score = score.long()
        if torch.cuda.is_available():
            score = torch.autograd.Variable(score.cuda())
        else:
            score = torch.autograd.Variable(score)
        beat_str = folk_dataset.get_beat_strength(score)
        rhy_ent = folk_dataset.get_rhythmic_entropy(score)
        #num_notes = folk_dataset.get_num_notes_in_measure(score)
        #note_range = folk_dataset.get_note_range_of_measure(score)
        sys.exit()
    #folk_dataset = FolkDataset('folk', cache_dir='../dataset_cache')
    # folk_dataset.download_raw_dataset()
    #folk_dataset.make_tensor_dataset()
    '''