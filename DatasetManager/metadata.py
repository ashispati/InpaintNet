"""
Metadata classes
"""
import numpy as np
from music21 import meter
from DatasetManager.helpers import SLUR_SYMBOL, \
    PAD_SYMBOL, BEAT_SYMBOL, DOWNBEAT_SYMBOL


class Metadata:
    def __init__(self):
        self.num_values = None
        self.is_global = None
        self.name = None

    def get_index(self, value):
        # trick with the 0 value
        raise NotImplementedError

    def get_value(self, index):
        raise NotImplementedError

    def evaluate(self, chorale, subdivision):
        """
        takes a music21 chorale as input and the number of subdivisions per beat
        """
        raise NotImplementedError

    def generate(self, length):
        raise NotImplementedError


class IsPlayingMetadata(Metadata):
    def __init__(self, voice_index, min_num_ticks):
        """
        Metadata that indicates if a voice is playing
        Voice i is considered to be muted if more than 'min_num_ticks' contiguous
        ticks contain a rest.


        :param voice_index: index of the voice to take into account
        :param min_num_ticks: minimum length in ticks for a rest to be taken
        into account in the metadata
        """
        super(IsPlayingMetadata, self).__init__()
        self.min_num_ticks = min_num_ticks
        self.voice_index = voice_index
        self.is_global = False
        self.num_values = 2
        self.name = 'isplaying'

    def get_index(self, value):
        return int(value)

    def get_value(self, index):
        return bool(index)

    def evaluate(self, chorale, subdivision):
        """
        takes a music21 chorale as input
        """
        length = int(chorale.duration.quarterLength * subdivision)
        metadatas = np.ones(shape=(length,))
        part = chorale.parts[self.voice_index]

        for note_or_rest in part.notesAndRests:
            is_playing = True
            if note_or_rest.isRest:
                if note_or_rest.quarterLength * subdivision >= self.min_num_ticks:
                    is_playing = False
            # these should be integer values
            start_tick = note_or_rest.offset * subdivision
            end_tick = start_tick + note_or_rest.quarterLength * subdivision
            metadatas[start_tick:end_tick] = self.get_index(is_playing)
        return metadatas

    def generate(self, length):
        return np.ones(shape=(length,))


class TickMetadata(Metadata):
    """
    Metadata class that tracks on which subdivision of the beat we are on
    """

    def __init__(self, subdivision):
        super(TickMetadata, self).__init__()
        self.is_global = False
        self.num_values = subdivision
        self.name = 'tick'

    def get_index(self, value):
        return value

    def get_value(self, index):
        return index

    def evaluate(self, chorale, subdivision):
        assert subdivision == self.num_values
        # suppose all pieces start on a beat
        length = int(chorale.duration.quarterLength * subdivision)
        return np.array(list(map(
            lambda x: x % self.num_values,
            range(length)
        )))

    def generate(self, length):
        return np.array(list(map(
            lambda x: x % self.num_values,
            range(length)
        )))


class BeatMarkerMetadata(Metadata):
    """
    Metadata class that tracks the beat and downbeat markers
    """
    def __init__(self, subdivision):
        super(BeatMarkerMetadata, self).__init__()
        self.is_global = False
        self.num_values = subdivision
        self.name = 'beatmarker'
        # create beatmarker dictionaries
        self.beat_index2symbol_dicts = {}
        self.beat_symbol2index_dicts = {}
        beat_set = set()
        beat_set.add(PAD_SYMBOL)    
        beat_set.add(SLUR_SYMBOL)
        beat_set.add(BEAT_SYMBOL)
        beat_set.add(DOWNBEAT_SYMBOL)
        for beat_index, beat in enumerate(beat_set):
            self.beat_index2symbol_dicts.update({beat_index: beat})
            self.beat_symbol2index_dicts.update({beat: beat_index})
        print(self.beat_index2symbol_dicts)

    def get_index(self, value):
        return value
    
    def get_value(self, index):
        return index 

    def evaluate(self, leadsheet, subdivision):
        assert subdivision == self.num_values
        # assume all pieces start on the downbeat
        symbol2index = self.beat_symbol2index_dicts
        # get time signature numerator (number of beats in a measure)
        ts = leadsheet.parts[0].recurse().getElementsByClass(meter.TimeSignature)
        if len(ts) == 1:
            beats_per_measure = ts[0].numerator
        else:
            beats_per_measure = 4
        assert(beats_per_measure == 3 or beats_per_measure == 4)
        freq = beats_per_measure * subdivision

        # find the length of the metadata tensor
        length = int(leadsheet.highestTime * subdivision)
        t = np.ones((1, length)) * symbol2index[SLUR_SYMBOL]
    
        # construct sequence
        t[0::freq] = symbol2index[DOWNBEAT_SYMBOL]
        t[0 + subdivision:: freq] = symbol2index[BEAT_SYMBOL]
        t[0 + 2 * subdivision:: freq] = symbol2index[BEAT_SYMBOL]
        if beats_per_measure == 4:
            t[0 + 3 * subdivision:: freq] = symbol2index[BEAT_SYMBOL]
        return t
    
    def generate(self, length):
        symbol2index = self.beat_symbol2index_dicts
        beats_per_measure = 4  # TODO: remove this hardcoding
        subdivision = self.num_values
        freq = beats_per_measure * subdivision
        t = np.ones((1, length)) * symbol2index[SLUR_SYMBOL]
    
        # construct sequence
        t[0::freq] = symbol2index[DOWNBEAT_SYMBOL]
        t[0 + subdivision:: freq] = symbol2index[BEAT_SYMBOL]
        t[0 + 2 * subdivision:: freq] = symbol2index[BEAT_SYMBOL]
        if beats_per_measure == 4:
            t[0 + 3 * subdivision:: freq] = symbol2index[BEAT_SYMBOL]
        return t
