
from typing import Any
import tensorflow as tf


class TensorBoardLogger():
    def __init__(self, log_dir: str):
        self._logger = tf.summary.create_file_writer(log_dir)
        self.step = 1

    def logging_scaler(self, data_label: str, data: Any) -> None:
        with self._logger.as_default():
            tf.summary.scalar(data_label, data, self.step)
        self.step += 1
