from g2p_seq2seq_pytorch.g2p import G2PPytorch
from utils import (
    dict_miss_words,
    process_tltchool_gigaspeech_interregnum_tokens
)


class G2PModel(object):
    def __init__(self, lexicon_file_path):
        # G2P related
        self.g2p_model = G2PPytorch()
        self.g2p_model.load_model()
        self.lexicon_file_path = lexicon_file_path

    def g2p(self, text_predicted, word2phn_dict):
        missed_words = dict_miss_words(text_predicted, word2phn_dict)
        missed_words = process_tltchool_gigaspeech_interregnum_tokens(missed_words)

        if missed_words == '':
            return word2phn_dict

        for word in missed_words.split():
            phns_string = self.g2p_model.decode_word(word)
            self._write_lexicon(self.lexicon_file_path, word, phns_string)
            word2phn_dict[word] = [p.lower() for p in phns_string.split()]
        return word2phn_dict

    def _write_lexicon(self, lexicon_file_path, word, phns_string):
        with open(lexicon_file_path, 'a') as f:
            f.write("{} {}\n".format(word, phns_string.lower()))