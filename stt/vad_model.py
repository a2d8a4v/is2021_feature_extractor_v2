#!/usr/bin/env python3
#
# This script gets speech segments from whole recordings using webrtcvad
# Modified from: https://github.com/wiseman/py-webrtcvad/blob/master/example.py
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

import argparse
import collections
import contextlib
import os
import sys
import wave
import webrtcvad
import soundfile
import numpy as np

class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

class VadModel(object):
    def __init__(self, mode=1, sample_rate=16000, frame_duration_ms=30):
        '''
        Integer in {0,1,2,3} specifying the VAD aggressiveness.
        0 is the least aggressive
        about filtering out non-speech, 3 is the most aggressive.
        '''
        if mode not in [0, 1, 2, 3]:
            raise Exception("Aggressiveness mode must be in {0,1,2,3}")
        else:
            self.mode = mode
        self.vad = webrtcvad.Vad(self.mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        
    def read_wave(self, path):
        """Reads a .wav file.
        Takes the path, and returns (PCM audio data, sample rate).
        """ 
        with contextlib.closing(wave.open(path, "rb")) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate == self.sample_rate
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate
    
    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset : offset + n], timestamp, duration)
            timestamp += duration
            offset += n
        
    def vad_segments(self, sample_rate, frame_duration_ms, padding_duration_ms, frames):
        """Filters out non-voiced audio frames.
        Given a webrtcvad.Vad and a source of audio frames, yields only
        the voiced audio.
        Uses a padded, sliding window algorithm over the audio frames.
        When more than 90% of the frames in the window are voiced (as
        reported by the VAD), the collector triggers and begins yielding
        audio frames. Then the collector waits until 90% of the frames in
        the window are unvoiced to detrigger.
        The window is padded at the front and back to provide a small
        amount of silence or the beginnings/endings of speech around the
        voiced frames.
        Arguments:
        sample_rate - The audio sample rate, in Hz.
        frame_duration_ms - The frame duration in milliseconds.
        padding_duration_ms - The amount to pad the window, in milliseconds.
        vad - An instance of webrtcvad.Vad.
        frames - a source of audio frames (sequence or generator).
        Returns: List of (start_time,end_time) tuples.
        """
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        triggered = False
        segments = []
        voiced_frames = []
        vad = self.vad
        
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    start_time = voiced_frames[0].timestamp
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    end_time = frame.timestamp + frame.duration
                    triggered = False
                    ring_buffer.clear()
                    voiced_frames = []
                    # Write to segments list
                    segments.append((start_time, end_time))
        # If we have any leftover voiced audio when we run out of input,
        # add it to segments list.
        if voiced_frames:
            end_time = voiced_frames[-1].timestamp
            segments.append((start_time, end_time))
        return segments
        
    def get_speech_segments(self, audio, sample_rate=16000):
        """
        Compute and print the segments for the given uttid. It is in the format:
        <segment-id> <utt-id> <start-time> <end-time>
        """
        frame_duration_ms = self.frame_duration_ms
        frames = self.frame_generator(frame_duration_ms, audio, sample_rate)
        frames = list(frames)
        segments = self.vad_segments(sample_rate, frame_duration_ms, 300, frames)
        speech = np.frombuffer(audio, dtype='int16').astype(np.float32) / 32768.0
        voiced_speechs = []
        
        for segment in segments:
            start = float("{:.2f}".format(segment[0]))
            end = float("{:.2f}".format(segment[1]))
            voiced_speech = speech[int(segment[0] * 16000): int(segment[1] * 16000)]
            voiced_speechs.append(voiced_speech)
        return voiced_speechs