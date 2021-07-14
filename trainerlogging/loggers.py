import csv
import io
import itertools
import os
from os import makedirs
from sys import path
from typing import Dict, Any, List, Union
from zipfile import ZipFile

import torch
from matplotlib import pyplot as plt


def _alpha_gen(alphabet=None):
    if alphabet is None:
        alphabet = "xyzwabcdefghijklmnopqrstuv"
    for size in itertools.count(1):
        for s in itertools.product(alphabet, repeat=size):
            yield "".join(s)


def _read_num(value: str):
    try:
        return int(value)
    except ValueError:
        return float(value)  # raises ValueError if failed


class DictionaryLogger:

    def __init__(self, name: str):
        self.name = name
        self._log: List[Dict[str, Any]] = [dict(step=0)]
        self._aux_logs: Dict[str, List[Dict[str, Any]]] = {}

    @classmethod
    def load_table(cls, csv_data) -> List[Dict[str, Any]]:
        table = []
        reader = csv.reader(csv_data)
        headers = next(reader)
        for row in reader:
            entry = {}
            table.append(entry)
            for h, c in zip(headers, row):
                try:
                    entry[h] = _read_num(c)
                except ValueError:
                    continue
        return table

    @classmethod
    def load(cls, file: str):
        directory, filename = path.split(file)
        logger_name, extension = filename[:-4], filename[-4:]
        logger = cls(logger_name)
        if extension == '.csv':
            with open(file, 'r', newline='') as csv_file:
                logger._log = logger.load_table(csv_file)
        elif extension == '.zip':
            with ZipFile(file) as zf:
                for f in zf.filelist:
                    table_name = f.filename[len(logger_name) + 2:-4]
                    table = DictionaryLogger.load_table(io.TextIOWrapper(zf.open(f)))
                    if table_name == 'main':
                        logger._log = table
                    else:
                        logger._aux_logs[table_name] = table
        else:
            return None
        return logger

    @staticmethod
    def _save_table(table: List[Dict[str, Any]]) -> str:
        cpu_table = [{col: v.numpy() if isinstance(v, torch.Tensor) else v for col, v in row.items()} for row in table]
        column_names = sorted({col for entry in cpu_table for col in entry.keys()})
        with io.StringIO() as buffer:
            writer = csv.DictWriter(buffer, fieldnames=column_names)
            writer.writeheader()
            writer.writerows(cpu_table)
            return buffer.getvalue()

    def write(self, step=None, *args, **kwargs):
        if step is None: # Update the state of current step
            self._log[-1].update(dict(*args, **kwargs))
            return
        if step >= len(self._log):
            self._log.append(dict(*args, **kwargs, step=len(self._log)))
        else:
            self._log[step].update(dict(*args, **kwargs))

    def write_table(self, name: str, *data, **kwargs):
        if len(data) == 0:
            return
        if len(data) > 1:
            data = zip(*data)
        else:
            data = data[0]
        self._aux_logs[name] = [dict({col: v for col, v in zip(_alpha_gen(), axes)}, **kwargs) for axes in data]

    def save(self, directory: str, overwrite: bool = True):
        try:
            makedirs(directory)
        except FileExistsError:
            pass
        zip_filepath = path.join(directory, f'{self.name}.zip')
        if overwrite and path.exists(zip_filepath):
            os.remove(zip_filepath)
        try:
            with ZipFile(zip_filepath, 'x') as f:
                f.writestr(f'{self.name}__main.csv', self._save_table(self._log))
                for aux_log_name, aux_log in self._aux_logs.items():
                    f.writestr(f'{self.name}__{aux_log_name}.csv', self._save_table(aux_log))
        except FileExistsError:
            pass

    def plot2D(self, ax: plt.Axes, y_axis: Union[str, List[str]], x_axis: str = 'step', table: Union[str, List[str]] = 'main'):
        if isinstance(y_axis, str):
            y_axis = [y_axis]
        if isinstance(table, str):
            table = [table]

        tables = ((t, self._aux_logs[t] if t != "main" else self._log) for t in table)

        for t_name, t in tables:
            for y_ in y_axis:
                print(t)
                x, y = zip(*((d.get(x_axis, 0), d[y_]) for d in t if y_ in d))
                ax.plot(x, y, markevery=2, label=f"{t_name}/{y_}")
        ax.legend()
