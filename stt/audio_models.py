from scipy.io import wavfile
from scipy.signal import correlate, fftconvolve, lfilter, hamming
from scipy.interpolate import interp1d
# from linpred import lpc

from PyToBI.tobi import TextGridOperations

import librosa

import tempfile
import os
import math
import numpy as np
import json
import soundfile
from tqdm import tqdm
import parselmouth
from utils import (
    run_praat,
    getLeft,
    getRight
)

'''
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
'''
def merge_dict(first_dict, second_dict):
    third_dict = {**first_dict, **second_dict}
    return third_dict

def get_stats(numeric_list, prefix=""):
    # number, mean, standard deviation (std), median, mean absolute deviation
    stats_np = np.array(numeric_list)
    number = len(stats_np)
    
    if number == 0:
        summ = 0.
        mean = 0.
        std = 0.
        median = 0.
        mad = 0.
        maximum = 0.
        minimum = 0.
    else:
        summ = np.float64(np.sum(stats_np))
        mean = np.float64(np.mean(stats_np))
        std = np.float64(np.std(stats_np))
        median = np.float64(np.median(stats_np))
        mad = np.float64(np.sum(np.absolute(stats_np - mean)) / number)
        maximum = np.float64(np.max(stats_np))
        minimum = np.float64(np.min(stats_np))
    
    stats_dict = {  prefix + "number": number, 
                    prefix + "mean": mean, 
                    prefix + "std": std, 
                    prefix + "median": median, 
                    prefix + "mad": mad, 
                    prefix + "summ": summ,
                    prefix + "max": maximum,
                    prefix + "min": minimum
                 }
    return stats_dict
    
    
class AudioModel(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.frame_length = 25 * self.sample_rate // 1000
    
    def get_f0(self, speech, frame_length=800, win_length=400):
        # it use ceil method to get the length of f0 list
        f0_list, voiced_flag, voiced_probs = librosa.pyin(speech, sr=self.sample_rate,
                                             frame_length=frame_length,
                                             win_length=win_length, hop_length=160, center=False, 
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'))
        f0_list = np.nan_to_num(f0_list)
        f0_stats = get_stats(f0_list, prefix="f0_")
        f0_stats["f0_list"] = f0_list.tolist()
        f0_stats["f0_voiced_probs"] = voiced_probs.tolist()
        # removed unvoiced frames
        f0_nz_list = f0_list[np.nonzero(f0_list)]
        f0_nz_stats = get_stats(f0_nz_list, prefix="f0_nz_")
        f0_nz_stats["f0_nz_list"] = f0_nz_list.tolist()
        f0_stats = merge_dict(f0_stats, f0_nz_stats)
        
        return [f0_list, f0_stats]
    
    def get_energy(self, speech, frame_length=800):
        # alignment (stt)
        rms = librosa.feature.rms(speech, frame_length=frame_length,
                                  hop_length=160, center=False)
        rms_list = rms.reshape(rms.shape[1],)
        rms_stats = get_stats(rms_list, prefix="energy_")
        rms_stats["energy_rms_list"] = rms_list.tolist()
        
        return [rms_list, rms_stats]

    def get_pitch(self, speech, ctm_info, f0_list, total_duration, formants_info=None, frame_length=200, win_length=100):
        pitch_list = []

        # NOTICE: this part is a big bottle-neck on time complexity, it is better to use formant info with timestamp info inside
        if formants_info is None:

            for word, start_time, duration, conf in ctm_info:
                start_time = math.ceil(start_time*self.sample_rate)
                end_time = math.ceil((start_time + duration)*self.sample_rate)
                _speech = speech[start_time:end_time]
                word_f0_list, _ = self.get_f0(_speech, frame_length=frame_length, win_length=win_length)
                del _
                pitch_list.append(sum(word_f0_list)/len(word_f0_list))

        else:

            formants_info = np.array(formants_info)

            for word, start_time, duration, conf in ctm_info:
                end_time = start_time + duration
                interval = [
                    getLeft(start_time, total_duration, formants_info),
                    getRight(end_time, total_duration, formants_info)
                ]
                word_f0_list = np.array(f0_list)[interval]
                pitch_list.append(np.float64(np.mean(word_f0_list)))

        return pitch_list
        
    def get_intensity(self, speech, ctm_info, energy_list, total_duration, formants_info=None, frame_length=200):
        intensity_list = []

        # NOTICE: this part is a big bottle-neck on time complexity, it is better to use formant info with timestamp info inside
        if formants_info is None:

            for word, start_time, duration, conf in ctm_info:
                start_time = math.ceil(start_time*self.sample_rate)
                end_time = math.ceil((start_time + duration)*self.sample_rate)
                _speech = speech[start_time:end_time]
                word_rms_list, _ = self.get_energy(_speech, frame_length=frame_length)
                del _
                intensity_list.append(sum(word_rms_list)/len(word_rms_list))

        else:

            formants_info = np.array(formants_info)

            for word, start_time, duration, conf in ctm_info:
                end_time = start_time + duration
                interval = [
                    getLeft(start_time, total_duration, formants_info),
                    getRight(end_time, total_duration, formants_info)
                ]
                word_energy_list = np.array(energy_list)[interval]
                intensity_list.append(np.float64(np.mean(word_energy_list)))

        return intensity_list

    def __make_script(self):
        # _script_loc = tempfile.mktemp(suffix='.praat')
        _script_loc = os.path.abspath('./_formants.praat')
        with open(_script_loc, 'w') as fid:
            fid.write("""# take name of wav file from stdin and dump formant table to stdout
form File
sentence filename
positive maxformant 5500
real winlen 0.025
positive preemph 50
endform
Read from file... 'filename$'
To Formant (burg)... 0.01 5 'maxformant' 'winlen' 'preemph'
List... no yes 6 no 3 no 3 no
exit""")
        return _script_loc

    def __file2formants(self, filename, maxformant=5500, winlen=0.025, preemph=50):
        """Extract formant table from audio file using praat.
        The return array is laid out as:
        [[time, f1, f2, f3],
        ..]

        Formants that praat returns as undefined are represented as NaNs. This
        function can be memoized to minimize the number of calls to praat.

        Arguments:
        :param filename: filename of audio file
        :param maxformant: formant ceiling (use 5500 for female speech, 5000 for male)
        :param winlen: window length in seconds [0.025]
        :param preemph: pre-emphasis [50]
        """
        def _float(s):
            try:
                return float(s)
            except ValueError:
                return np.nan

        """
        # This will print the info from the Fromant object, and res[0] is a parselmouth.Fromant object
        # This will print the info from the textgrid object, and res[1] is a parselmouth.Data object with a TextGrid inside
        """
        # _dir = os.path.dirname(os.path.abspath(__file__))
        # formant_script_path = os.path.join(_dir, "formant/formants.praat")
        # print(formant_script_path)
        # res = run_praat(formant_script_path, filename, maxformant, winlen, preemph)[1]
        res = run_praat(self.__make_script(), filename, maxformant, winlen, preemph)[1]
        rtn = np.array([list(map(_float, x.strip().split('\t')[:6])) for x in res.split('\n')[1:-1]]).tolist()
        return rtn

    def __lpc_formants(self, speech, sample_rate):

        # Get speech data
        x = speech

        # Get Hamming window.
        N = len(x)
        w = np.hamming(N)

        # Apply window and high pass filter.
        x1 = x * w
        x1 = lfilter([1], [1., 0.63], x1)

        # Get LPC.
        ncoeff = 2 + sample_rate / 1000
        # BUG: Cannot find lpc function
        A, e, k = lpc(x1, ncoeff)

        # Get roots.
        rts = np.roots(A)
        rts = [r for r in rts if np.imag(r) >= 0]

        # Get angles.
        angz = np.arctan2(np.imag(rts), np.real(rts))

        # Get frequencies.
        frqs = sorted(angz * (sample_rate / (2 * math.pi)))
        print("frqs: {}".format(frqs))
        input()

        return frqs

    def get_formants(self, speech_path, speech, sample_rate, method="praat"):
        """
        References:
            [1] http://blog.syntheticspeech.de/2021/03/10/how-to-extract-formant-tracks-with-praat-and-python/
            [2] https://stackoverflow.com/questions/25107806/estimate-formants-using-lpc-in-python#
            [3] https://github.com/mwv/praat_formants_python/blob/master/praat_formants_python/_formants.py

        Formant Frequency: To illustrate vocal tract shapes during the production fo the vowels, 
        the first formant (F1) inversely correlates with tongue height, and the second formant (F2) with tongue backness.

        Return:
            List: [
                [t0, F1_0, F2_0, F3_0, F4_0, F5_0],
                [t1, F1_1, F2_1, F3_1, F4_1, F5_1],
                ...
            ]
        """

        assert method in ["praat", "lpc"], "method should only have praat or lpc two methods"

        if method == "praat":
            return self.__file2formants(speech_path)
        elif method == "lpc":
            return self.__lpc_formants(speech, sample_rate)

        return None

    def get_tobi(self, speech_path, tobi_path, textgrid_file_path):
        """
        :File path should be absolute path
        """

        # ('speakerIp16_A1_001001001001-promptIp16_A1_en_5_53_101', '.wav')
        basename_wo_ext = os.path.splitext(os.path.basename(speech_path))[0]

        _dir = os.path.dirname(os.path.abspath(__file__))
        _ = run_praat(
            os.path.join(_dir, "PyToBI/praatScripts/module01.praat"),
            speech_path,
            tobi_path,
            basename_wo_ext,
            capture_output=False
        )
        _ = run_praat(
            os.path.join(_dir, "PyToBI/praatScripts/module02.praat"),
            tobi_path,
            basename_wo_ext,
            capture_output=False
        )
        _ = run_praat(
            os.path.join(_dir, "PyToBI/praatScripts/module03.praat"),
            tobi_path,
            basename_wo_ext,
            capture_output=False
        )
        _ = run_praat(
            os.path.join(_dir, "PyToBI/praatScripts/module04.praat"),
            tobi_path,
            basename_wo_ext,
            textgrid_file_path,
            capture_output=False
        )

        pathIn = os.path.join(tobi_path, basename_wo_ext + "_mod4.TextGrid") 
        pathOut = os.path.join(tobi_path, basename_wo_ext + "_result.TextGrid") 
        _ = TextGridOperations(pathIn, pathOut)
        del _

        return pathOut

if __name__ == "__main__":
    import soundfile
    wav_path = "data/spoken_test_2022_jan28/wavs/0910102838/0910102838-2-6-2022_1_13.wav"
    speech, rate = soundfile.read(wav_path)
    assert rate == 16000
    
    audio_model = AudioModel()
    _, f0_info, _, f0_nz_info = audio_model.get_f0(speech)
