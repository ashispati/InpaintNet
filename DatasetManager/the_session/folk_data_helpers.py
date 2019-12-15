import os
import numpy as np

from tqdm import tqdm
from glob2 import glob
from fractions import Fraction
from music21 import meter
from music21.abcFormat import ABCHandlerException

from DatasetManager.helpers import *

# dictionary
note_values = {
    'q':  1.,
    'h':  2.,
    'w':  4.,
    '8':  0.5,
    '16': 0.25,
    '32': 0.125
}

tick_values = [
    0,
    Fraction(1, 4),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(2, 3),
    Fraction(3, 4)
]

MAX_NOTES = 140


class FakeNote:
    """
    Class used to have SLUR_SYMBOLS with a duration
    """

    def __init__(self, symbol, duration):
        self.symbol = symbol
        self.duration = duration

    def __repr__(self):
        return f'<FakeNote {self.symbol}>'


def score_on_ticks(score, tick_vals):
    notes, _ = notes_and_chords(score)
    eps = 1e-5
    for n in notes:
        _, d = divmod(n.offset, 1)
        flag = False
        for tick_value in tick_vals:
            if tick_value - eps < d < tick_value + eps:
                flag = True
        if not flag:
            return False

    return True


def get_notes_in_measure(measure):
    """
    Returns the notes in a music21 measure object
    :param measure: music21 measure object
    :return:
    """
    notes = measure.flat.notesAndRests
    notes = [n for n in notes if not isinstance(n, music21.harmony.ChordSymbol)]
    return notes


def notes_and_chords(score):
    """
    Returns the notes and chords from the music21 score object
    :param score: music21 score
    :return:
    """
    notes = score.parts[0].flat.notesAndRests
    notes = [n for n in notes if not isinstance(n, music21.harmony.ChordSymbol)]
    chords = score.parts[0].flat.getElementsByClass(
        [music21.harmony.ChordSymbol,
         music21.expressions.TextExpression
         ])
    return notes, chords


def get_notes(score):
    """
    Returns the notes from the music21 score object
    :param score: music21 score
    :return:
    """
    s = score.parts[0]
    # score.show()

    measures = s.recurse().getElementsByClass(music21.stream.Measure)
    # check for pick-up measures
    if measures[0].barDurationProportion() != 1.0:
        offset = measures[0].paddingLeft
        measures[0].insertAndShift(
            0.0, music21.note.Rest(quarterLength=offset))
    for m in measures:
        notes = get_notes_in_measure(m)
    notes = s.flat.notesAndRests
    notes = [n for n in notes if not isinstance(
        n, music21.harmony.ChordSymbol)]
    return notes


def score_range(score):
    """

    :param score: music21 score object
    :return: tuple int, min and max midi pitch numbers
    """
    notes, _ = notes_and_chords(score)
    pitches = [n.pitch.midi for n in notes if n.isNote]
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    return min_pitch, max_pitch


class FolkIteratorGenerator:
    """
    Object that returns a iterator over folk dataset (as music21 scores)
    when called
    :return:
    """

    def __init__(
            self,
            num_elements=None,
            time_sigs=[(4, 4)],
            has_key=False
    ):
        """
        :param num_elements: int, number of tunes to be considered
        :param time_sigs: list of tuples, each tuple should specify
                          the allowed time signatures, with the first 
                          element being the time signature numerator and
                          the second element being the time signature
                          denominator 
        """
        self.package_dir = os.path.dirname(os.path.realpath(__file__))
        self.raw_dataset_dir = os.path.join(
            self.package_dir,
            'raw_data',
        )

        self.raw_dataset_url = 'https://raw.githubusercontent.com/IraKorshunova/' \
                               'folk-rnn/master/data/' \
                               'sessions_data_clean.txt'

        if not os.path.exists(self.raw_dataset_dir):
            os.mkdir(self.raw_dataset_dir)

        self.full_raw_dataset_filepath = os.path.join(
            self.raw_dataset_dir,
            'raw_dataset_full.txt'
        )

        if not os.path.exists(self.full_raw_dataset_filepath):
            self.download_raw_dataset()
            self.split_raw_dataset()

        self.has_chords = False
        self.has_key = has_key
        if time_sigs is None:
            self.time_sigs = [(4, 4)]    # 4by4 is the default time signature
        else:
            self.time_sigs = time_sigs

        self.valid_files_list = os.path.join(
            self.package_dir,
            self.__repr__() + 'valid_filepaths.txt'
        )
        
        # read and store the valid file paths
        self.valid_tune_filenames = []
        self.valid_file_indices = []
        
        # set num_elements for the iterator
        if num_elements is None:
            self.num_elements = 25000
        else:
            self.num_elements = num_elements

    def __repr__(self):
        if self.has_chords:
            chord_str = 'chords'
        else:
            chord_str = ''
        if self.has_key:
            key_str = 'key'
        else:
            key_str = ''
        name_str = 'FolkItGen(' + \
                   key_str + \
                   chord_str + \
                   str(self.time_sigs).replace(" ", "") + ')'
        return name_str

    def download_raw_dataset(self):
        if os.path.exists(self.full_raw_dataset_filepath):
            print('The Session dump already exists')
        else:
            print('Downloading The Session dump')
            os.system(
                f'wget -L {self.raw_dataset_url} -O {self.full_raw_dataset_filepath}')

    def split_raw_dataset(self):
        print('Splitting raw dataset')
        with open(self.full_raw_dataset_filepath) as full_raw_dataset_file:
            tune_index = 0
            current_song_filepath = os.path.join(self.raw_dataset_dir,
                                                 f'tune_{tune_index}.abc')
            current_song_file = open(current_song_filepath, 'w+')
            for line in full_raw_dataset_file:
                if line == '\n':
                    tune_index += 1
                    current_song_file.flush()
                    current_song_file.close()
                    current_song_filepath = os.path.join(self.raw_dataset_dir,
                                                         f'tune_{tune_index}.abc')
                    current_song_file = open(current_song_filepath, 'w+')
                else:
                    current_song_file.write(line)

    def __call__(self, *args, **kwargs):
        it = (
            score
            for score in self.score_generator()
        )
        return it

    def score_generator(self):
        self.get_valid_tune_filepaths()
        for score_index, score_path in enumerate(self.valid_tune_filenames):
            if score_index >= self.num_elements:
                continue
            try:
                full_path = os.path.join(self.raw_dataset_dir, score_path)
                yield self.get_score_from_path(full_path, fix_and_expand=True)
            except ZeroDivisionError:
                print(f'{score_path} is not parsable')

    def get_valid_tune_filepaths(self):
        """
        Stores a list of filepaths for all valid tunes in dataset
        """
        if os.path.exists(self.valid_files_list):
            print('List already exists. Reading it now')
            f = open(self.valid_files_list, 'r')
            self.valid_tune_filenames = [line.rstrip('\n') for line in f]
            print(f'Number of file: {len(self.valid_tune_filenames)}')
            return

        print('Checking dataset for valid files')
        tune_filepaths = glob(f'{self.raw_dataset_dir}/tune*')
        self.valid_file_indices = []
        self.valid_tune_filenames = []
        # get list of allowed time signatures
        ts_num_allowed = []
        ts_den_allowed = []
        for ts in self.time_sigs:
            ts_num, ts_den = ts
            ts_num_allowed.append(ts_num)
            ts_den_allowed.append(ts_den)
        for tune_index, tune_filepath in tqdm(enumerate(tune_filepaths)):
            title = self.get_title(tune_filepath)
            # ignore files without titles
            if title is None:
                continue
            # ignore files with multiple voices
            if self.tune_is_multivoice(tune_filepath):
                continue
            # ignore files with chords
            if self.tune_contains_chords(tune_filepath):
                continue
            # handle key
            if self.has_key:
                if self.get_key(tune_filepath) is None:
                    continue
            try:
                score = self.get_score_from_path(tune_filepath)
                # ignore files with not allowed time signatures
                ts = score.parts[0].recurse().getElementsByClass(meter.TimeSignature)
                if len(ts) > 1:
                    continue
                else:
                    ts_num = ts[0].numerator
                    ts_den = ts[0].denominator
                    if ts_den not in ts_den_allowed:
                        continue
                    else:
                        if ts_num not in ts_num_allowed:
                            continue
                        else:
                            # ignore files with no notes
                            notes, chords = notes_and_chords(score)
                            pitches = [n.pitch.midi for n in notes if n.isNote]
                            if len(pitches) == 0:
                                continue
                            # ignore files with too few or too high notes
                            if len(notes) > MAX_NOTES:
                                continue
                            # ignore files with 32nd and 64th notes
                            dur_list = [n.duration for n in notes if n.isNote]
                            for dur in dur_list:
                                d = dur.type
                                if d == '32nd':
                                    break
                                elif d == '64th':
                                    break
                                elif d == 'complex':
                                    # TODO: bad hack. fix this !!!
                                    if len(dur.components) > 2:
                                        break
                            # check if expand repeat works
                            score = self.get_score_from_path(tune_filepath, fix_and_expand=True)
                            # ignore files where notes are not on ticks
                            if not score_on_ticks(score, tick_values):
                                continue
                            else:
                                # add to valid tunes list
                                self.valid_file_indices.append(tune_index)
                                file_name = os.path.basename(tune_filepath)
                                self.valid_tune_filenames.append(file_name)
            except (music21.abcFormat.ABCHandlerException,
                    music21.abcFormat.ABCTokenException,
                    music21.duration.DurationException,
                    music21.pitch.AccidentalException,
                    music21.meter.MeterException,
                    music21.repeat.ExpanderException,
                    music21.exceptions21.StreamException,
                    music21.pitch.PitchException,
                    AttributeError,
                    IndexError,
                    UnboundLocalError,
                    ValueError,
                    ABCHandlerException) as e:
                print('Error when parsing ABC file: ', tune_index)
                print(e)

        f = open(self.valid_files_list, 'w')
        for tune_filepath in self.valid_tune_filenames:
            f.write("%s\n" % tune_filepath)
        f.close()

    def get_score_from_path(self, tune_filepath, fix_and_expand=False):
        """
        Extract music21 score from provided path to the tune

        :param tune_filepath: path to tune in .abc format
        :param fix_and_expand: bool, fix the pick-up measure offset and expand repeats if True
        :return: music21 score object
        """
        score = music21.converter.parse(tune_filepath, format='abc')
        if fix_and_expand:
            score = score.expandRepeats()
            score = self.fix_pick_up_measure_offset(score)
            score = self.fix_last_measure(score)
        return score

    def scan_dataset(self):
        # fix this
        num_files = len(self.valid_tune_filenames)
        num_notes = np.zeros(num_files, dtype=int)
        num_4_4 = 0
        num_3_4 = 0
        num_6_8 = 0
        num_other_ts = 0
        num_multi_ts = 0
        pitch_dist = np.zeros(128)
        min_pitch = 127
        max_pitch = 0
        dur_dist = np.zeros(8)
        num_fast_note_files = 0

        for i in tqdm(range(num_files)):
            tune_filepath = self.valid_tune_filenames[i]
            score = self.get_score_from_path(tune_filepath)

            fast_note_flag = False
            # get number of notes, pitch range and distribution
            notes, _ = notes_and_chords(score)
            pitches = [n.pitch.midi for n in notes if n.isNote]
            if not pitches:
                continue
            num_notes[i] = len(notes)
            min_p = min(pitches)
            max_p = max(pitches)
            if min_p < min_pitch:
                min_pitch = min_p
            if max_p > max_pitch:
                max_pitch = max_p
            for p in pitches:
                pitch_dist[p] += 1

            # get duration distribution
            dur_list = [n.duration for n in notes if n.isNote]
            for dur in dur_list:
                d = dur.type
                if d == 'quarter':
                    dur_dist[0] += 1
                elif d == 'eighth':
                    dur_dist[1] += 1
                elif d == 'half':
                    dur_dist[2] += 1
                elif d == '16th':
                    dur_dist[3] += 1
                elif d == 'whole':
                    dur_dist[4] += 1
                elif d == '32nd':
                    dur_dist[5] += 1
                if not fast_note_flag:
                    num_fast_note_files += 1
                    fast_note_flag = True
                elif d == '64th':
                    dur_dist[6] += 1
                if not fast_note_flag:
                    num_fast_note_files += 1
                    fast_note_flag = True
                else:
                    if d == 'complex':
                        dur_dist[7] += 1
                        print('**')
                        print(dur.components)
                        if not fast_note_flag:
                            num_fast_note_files += 1
                            fast_note_flag = True

            # get time signature
            ts = score.parts[0].recurse().getElementsByClass(meter.TimeSignature)
            if len(ts) > 1:
                num_multi_ts += 1
            else:
                ts_num = ts[0].numerator
                ts_den = ts[0].denominator
                if ts_den == 4:
                    if ts_num == 4:
                        num_4_4 += 1
                    elif ts_num == 3:
                        num_3_4 += 1
                    else:
                        num_other_ts += 1
                elif ts_den == 8:
                    if ts_num == 6:
                        num_6_8 += 1
                    else:
                        num_other_ts += 1
        print(f'Num Files: {num_files}')
        print(f'4/4: {num_4_4}')
        print(f'3/4: {num_3_4}')
        print(f'6/8: {num_6_8}')
        print(f'Others: {num_other_ts}')
        print(f'Multi: {num_multi_ts}')
        print(f'Min and Max Pitch: {min_pitch, max_pitch}')
        print(f'Num files with complex notes: {num_fast_note_files}')
        return num_notes, pitch_dist, dur_dist

    @staticmethod
    def fix_pick_up_measure_offset(score):
        """
        Adds rests to the pick-up measure (if-any)

        :param score: music21 score object
        """
        measures = score.recurse().getElementsByClass(music21.stream.Measure)
        num_measures = len(measures)
        # add rests in pick-up measures
        if num_measures > 0:
            m0_dur = measures[0].barDurationProportion()
            m1_dur = measures[1].barDurationProportion()
            if m0_dur != 1.0:
                if m0_dur + m1_dur != 1.0:
                    offset = measures[0].paddingLeft
                    measures[0].insertAndShift(0.0, music21.note.Rest(quarterLength=offset))
                    for i, m in enumerate(measures):
                        # shift the offset of all other measures
                        if i != 0:
                            m.offset += offset
        return score

    @staticmethod
    def fix_last_measure(score):
        """
        Adds rests to the last measure (if-needed)

        :param score: music21 score object
        """
        measures = score.recurse().getElementsByClass(music21.stream.Measure)
        num_measures = len(measures)
        # add rests in pick-up measures
        if num_measures > 0:
            m0_dur = measures[num_measures-1].barDurationProportion()
            if m0_dur != 1.0:
                offset = measures[num_measures-1].paddingRight
                measures[num_measures - 1].append(music21.note.Rest(quarterLength=offset))
        return score

    @staticmethod
    def get_title(tune_filepath):
        """

        :param tune_filepath: full path to the abc tune
        :return: str, title of the abc tune
        """
        for line in open(tune_filepath):
            if line[:2] == 'T:':
                return line[2:]
        return None

    @staticmethod
    def get_key(tune_filepath):
        """
        :param tune_filepath: full path to the abd tune
        :return str, key of the abc tune
        """
        count = 0
        key_str = None
        for line in open(tune_filepath):
            if line[:2] == 'K:':
                key_str = line[2:]
                count += 1
        if count == 1:
            return key_str
        else:
            return None

    @staticmethod
    def tune_contains_chords(tune_filepath):
        """

        :param tune_filepath: full path to the abc tune
        :return: bool, True if tune contains chords
        """
        for line in open(tune_filepath):
            if '"' in line:
                return True
        return False

    @staticmethod
    def tune_is_multivoice(tune_filepath):
        """

        :param tune_filepath: full path to the abc tune
        :return: bool, True if tune has mutiple voices
        """
        for line in open(tune_filepath):
            if line[:3] == 'V:2':
                return True
            if line[:4] == 'V: 2':
                return True
            if line[:4] == 'V :2':
                return True
            if line[:5] == 'V : 2':
                return True
        return False
