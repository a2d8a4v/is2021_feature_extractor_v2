
class Error(Exception):
    pass

class Which(object):

    # @https://hackage.haskell.org/package/hsc3-lang-0.15/docs/src/Sound-SC3-Lang-Data-CMUdict.html
    def __init__(self):
        self.consonants = ['l', 'zh', 's', 'z', 'ng', 'g', 'k', 'th', 'd', 'dh', 'w', 'p', 'n', 't', 'r', 'sh', 'ch', 'hh', 'b', 'jh', 'f', 'm', 'v']
        self.vowels = ['ah', 'aa', 'ih', 'aw', 'w', 'axr', 'ow', 'ao', 'y', 'eh', 'ay', 'uh', 'q', 'ey', 'ae', 'iy', 'oy', 'uw', 'ax', 'er']
        self.consonant = "C"
        self.vowel = "V"
        self.other = "O"

    def _is(self, ph):
        ph = ph.lower()
        if ph in self.vowels:
            return self.vowel
        elif ph in self.consonants:
            return self.consonant
        else:
            return self.other

    def get_v(self):
        return self.vowel

    def get_c(self):
        return self.consonant

    def get_o(self):
        return self.other

class Interval(object):

    def __init__(self):
        self.start = None
        self.end = None
        self.duration = None
        self.label = None
        self.type = None
    
    def set_start(self, s):
        self.start = s
        if self.end is not None:
            self.duration = self.end - self.start

    def set_end(self, e):
        self.end = e
        if self.start is not None:
            self.duration = self.end - self.start

    def set_label(self, l):
        self.label = l

    def set_type(self, t):
        self.type = t

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_label(self):
        return self.label

    def get_dur(self):
        return self.duration

    def get_type(self):
        return self.type