class TieException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ParsingException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class LeadsheetParsingException(ParsingException):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
