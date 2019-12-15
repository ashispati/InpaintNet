from music21 import note, harmony, expressions

# constants
SLUR_SYMBOL = '__'
START_SYMBOL = 'START'
END_SYMBOL = 'END'
OUT_OF_RANGE = 'OOR'
PAD_SYMBOL = 'XX'
BEAT_SYMBOL = 'b'
DOWNBEAT_SYMBOL = 'B'


def standard_name(note_or_rest, voice_range=None):
    """
    Convert music21 objects to str
    :param note_or_rest:
    :param voice_range:
    :return:
    """
    if isinstance(note_or_rest, note.Note):
        if voice_range is not None:
            min_pitch, max_pitch = voice_range
            pitch = note_or_rest.pitch.midi
            if pitch < min_pitch or pitch > max_pitch:
                return OUT_OF_RANGE
        return note_or_rest.nameWithOctave
    if isinstance(note_or_rest, note.Rest):
        return note_or_rest.name
    if isinstance(note_or_rest, str):
        return note_or_rest

    if isinstance(note_or_rest, harmony.ChordSymbol):
        return note_or_rest.figure
    if isinstance(note_or_rest, expressions.TextExpression):
        return note_or_rest.content


def standard_note(note_or_rest_string):
    if note_or_rest_string == 'rest':
        return note.Rest()
    # treat other additional symbols as rests
    elif (note_or_rest_string == END_SYMBOL
          or
          note_or_rest_string == START_SYMBOL
          or
          note_or_rest_string == PAD_SYMBOL):
        # print('Warning: Special symbol is used in standard_note')
        return note.Rest()
    elif note_or_rest_string == SLUR_SYMBOL:
        # print('Warning: SLUR_SYMBOL used in standard_note')
        return note.Rest()
    elif note_or_rest_string == OUT_OF_RANGE:
        # print('Warning: OUT_OF_RANGE used in standard_note')
        return note.Rest()
    else:
        return note.Note(note_or_rest_string)
