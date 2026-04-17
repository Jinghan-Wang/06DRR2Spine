import struct

import numpy as np


class PrjImage:
    dtype = np.uint16
    dtype2type = np.float32
    header_prefix_length = 3
    header_skip_length = 4
    cols_offset = header_prefix_length + header_skip_length
    rows_offset = cols_offset + 4

    def __init__(self):
        self.cols = 0
        self.rows = 0
        self.header_length = 1024
        self.full_path = None
        self.raw_header_bytes = None
        self.image_buffer = None

    @classmethod
    def to_array(cls, obj):
        return obj.get_image_data()

    @classmethod
    def read_ieee(cls, file, fmt):
        size = struct.calcsize(fmt)
        data = file.read(size)
        if not data:
            return None
        return struct.unpack(fmt, data)[0]

    @classmethod
    def read_int(cls, file):
        return cls.read_ieee(file, ">i")

    @classmethod
    def parse_header(cls, header_bytes):
        cols = struct.unpack_from(">i", header_bytes, cls.cols_offset)[0]
        rows = struct.unpack_from(">i", header_bytes, cls.rows_offset)[0]
        return cols, rows

    def set_buffer_size(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.image_buffer = np.zeros((rows, cols), dtype=PrjImage.dtype)

    def get_image_data(self):
        return self.image_buffer

    def get_size(self):
        return (self.cols, self.rows)

    @staticmethod
    def normalize(image_array, window_center=None, window_width=None):
        if window_center is not None and window_width is not None:
            win_min = window_center - window_width / 2
            win_max = window_center + window_width / 2
            img_float = image_array.astype(PrjImage.dtype2type, copy=False)
            np.clip(img_float, win_min, win_max, out=img_float)
            img_float -= win_min
            img_float /= window_width
            return img_float

        img_float = image_array.astype(PrjImage.dtype2type, copy=False)
        img_min = np.min(img_float)
        img_max = np.max(img_float)
        if img_max > img_min:
            img_float -= img_min
            img_float /= (img_max - img_min)
        else:
            img_float -= img_min
        return img_float

    def open(self, file_path, header_only=False, debug_flag=False):
        self.full_path = file_path
        with open(file_path, "rb") as f:
            self.raw_header_bytes = f.read(self.header_length)
            self.cols, self.rows = self.parse_header(self.raw_header_bytes)

            if debug_flag:
                self.printf_infos()

            if header_only:
                return self.raw_header_bytes

            f.seek(self.header_length, 0)
            data_size = self.cols * self.rows
            self.image_buffer = np.fromfile(f, dtype=PrjImage.dtype, count=data_size).reshape((self.rows, self.cols))
            array = self.normalize(self.image_buffer)

        return array

    def printf_infos(self):
        infos = "read_path:{}, dtype:{}, size:({},{})"
        print(infos.format(self.full_path, self.dtype, self.cols, self.rows))

    def extract_header(self):
        if self.raw_header_bytes is not None:
            return self.raw_header_bytes

        if self.full_path is None:
            raise ValueError("File path is not set. Call open() first or set full_path.")

        with open(self.full_path, "rb") as f:
            self.raw_header_bytes = f.read(self.header_length)

        return self.raw_header_bytes
