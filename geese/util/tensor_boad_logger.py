
from typing import Any
import tensorflow as tf


class TensorBoardLogger():
    def __init__(self, log_dir: str):
        self._logger = tf.summary.create_file_writer(log_dir)
        self._step_dict = {}

    def logging_scaler(self, data_label: str, data: Any) -> None:
        step = self._step_dict.get(data_label, 1)
        with self._logger.as_default():
            tf.summary.scalar(data_label, data, step)
        self._step_dict[data_label] = step+1
