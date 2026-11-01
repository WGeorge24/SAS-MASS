'''
Reader for EDF+ files (from DeepSleepNet / Boris Reuderink).
Used to parse PhysioNet hypnogram EDF files.
'''

import re
import datetime
import operator
import logging
import numpy as np
from collections import namedtuple
from functools import reduce

EVENT_CHANNEL = 'EDF Annotations'
log = logging.getLogger(__name__)


class EDFEndOfData(BaseException):
    pass


def tal(tal_str):
    exp = (
        r'(?P<onset>[+\-]\d+(?:\.\d*)?)'
        r'(?:\x15(?P<duration>\d+(?:\.\d*)?))?'
        r'(\x14(?P<annotation>[^\x00]*))?'
        r'(?:\x14\x00)'
    )

    def annotation_to_list(annotation):
        return annotation.split('\x14') if annotation else []

    def parse(dic):
        return (
            float(dic['onset']),
            float(dic['duration']) if dic['duration'] else 0.,
            annotation_to_list(dic['annotation']),
        )

    return [parse(m.groupdict()) for m in re.finditer(exp, tal_str)]


def edf_header(f):
    h = {}
    assert f.tell() == 0
    assert f.read(8) == '0       '

    h['local_subject_id'] = f.read(80).strip()
    h['local_recording_id'] = f.read(80).strip()

    (day, month, year) = [int(x) for x in re.findall(r'(\d+)', f.read(8))]
    (hour, minute, sec) = [int(x) for x in re.findall(r'(\d+)', f.read(8))]
    h['date_time'] = str(datetime.datetime(year + 2000, month, day, hour, minute, sec))

    header_nbytes = int(f.read(8))
    subtype = f.read(44)[:5]
    h['EDF+'] = subtype in ['EDF+C', 'EDF+D']
    h['contiguous'] = subtype != 'EDF+D'
    h['n_records'] = int(f.read(8))
    h['record_length'] = float(f.read(8))
    nchannels = h['n_channels'] = int(f.read(4))

    channels = list(range(h['n_channels']))
    h['label'] = [f.read(16).strip() for _ in channels]
    h['transducer_type'] = [f.read(80).strip() for _ in channels]
    h['units'] = [f.read(8).strip() for _ in channels]
    h['physical_min'] = np.asarray([float(f.read(8)) for _ in channels])
    h['physical_max'] = np.asarray([float(f.read(8)) for _ in channels])
    h['digital_min'] = np.asarray([float(f.read(8)) for _ in channels])
    h['digital_max'] = np.asarray([float(f.read(8)) for _ in channels])
    h['prefiltering'] = [f.read(80).strip() for _ in channels]
    h['n_samples_per_record'] = [int(f.read(8)) for _ in channels]
    f.read(32 * nchannels)

    assert f.tell() == header_nbytes
    return h


class BaseEDFReader:
    def __init__(self, file):
        self.file = file

    def read_header(self):
        self.header = h = edf_header(self.file)
        self.dig_min = h['digital_min']
        self.phys_min = h['physical_min']
        phys_range = h['physical_max'] - h['physical_min']
        dig_range = h['digital_max'] - h['digital_min']
        assert np.all(phys_range > 0)
        assert np.all(dig_range > 0)
        self.gain = phys_range / dig_range

    def read_raw_record(self):
        result = []
        for nsamp in self.header['n_samples_per_record']:
            samples = self.file.read(nsamp * 2)
            if len(samples) != nsamp * 2:
                raise EDFEndOfData
            result.append(samples)
        return result

    def convert_record(self, raw_record):
        h = self.header
        dig_min, phys_min, gain = self.dig_min, self.phys_min, self.gain
        time = float('nan')
        signals = []
        events = []
        for i, samples in enumerate(raw_record):
            if h['label'][i] == EVENT_CHANNEL:
                ann = tal(samples)
                time = ann[0][0]
                events.extend(ann[1:])
            else:
                dig = np.frombuffer(samples, '<i2').astype(np.float32)
                phys = (dig - dig_min[i]) * gain[i] + phys_min[i]
                signals.append(phys)
        return time, signals, events

    def read_record(self):
        return self.convert_record(self.read_raw_record())

    def records(self):
        try:
            while True:
                yield self.read_record()
        except EDFEndOfData:
            pass
