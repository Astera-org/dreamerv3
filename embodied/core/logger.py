import collections
import concurrent.futures
import json
import re
from pathlib import Path
from PIL import Image

import numpy as np

from ._encode_gif import encode_gif
from . import path
from . import printing
from . import timer

class Logger:

  def __init__(self, step, outputs, multiplier=1):
    assert outputs, 'Provide a list of logger outputs.'
    self.step = step
    self.outputs = outputs
    self.multiplier = multiplier
    self._last_step = None
    self._last_time = None
    self._metrics = []

  @timer.section('logger_add')
  def add(self, mapping, prefix=None):
    mapping = dict(mapping)
    # print('logger add:', len(mapping))
    assert len(mapping) <= 1000, list(mapping.keys())
    for key in mapping.keys():
      assert len(key) <= 200, (len(key), key[:200] + '...')
    step = int(self.step) * self.multiplier
    for name, value in mapping.items():
      name = f'{prefix}/{name}' if prefix else name
      if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, str):
        value = str(value)
      if not isinstance(value, str):
        value = np.asarray(value)
        if len(value.shape) not in (0, 1, 2, 3, 4):
          raise ValueError(
              f"Shape {value.shape} for name '{name}' cannot be "
              "interpreted as scalar, vector, image, or video.")
      self._metrics.append((step, name, value))

  def scalar(self, name, value):
    value = np.asarray(value)
    assert len(value.shape) == 0, value.shape
    self.add({name: value})

  def vector(self, name, value):
    value = np.asarray(value)
    assert len(value.shape) == 1, value.shape
    self.add({name: value})

  def image(self, name, value):
    value = np.asarray(value)
    assert len(value.shape) in (2, 3), value.shape
    self.add({name: value})

  def video(self, name, value):
    value = np.asarray(value)
    assert len(value.shape) == 4, value.shape
    self.add({name: value})

  def text(self, name, value):
    assert isinstance(value, str), (type(value), str(value)[:100])
    self.add({name: value})

  @timer.section('logger_write')
  def write(self):
    if not self._metrics:
      return
    for output in self.outputs:
      output(tuple(self._metrics))
    self._metrics.clear()

  def close(self):
    self.write()
    for output in self.outputs:
      if hasattr(output, 'wait'):
        try:
          output.wait()
        except Exception as e:
          print(f'Error waiting on output: {e}')


class AsyncOutput:

  def __init__(self, callback, parallel=True):
    self._callback = callback
    self._parallel = parallel
    if parallel:
      name = type(self).__name__
      self._worker = concurrent.futures.ThreadPoolExecutor(
          1, f'logger_{name}_async')
      self._future = None

  def wait(self):
    if self._parallel and self._future:
      concurrent.futures.wait([self._future])

  def __call__(self, summaries):
    if self._parallel:
      self._future and self._future.result()
      self._future = self._worker.submit(self._callback, summaries)
    else:
      self._callback(summaries)


class TerminalOutput:

  def __init__(self, pattern=r'.*', name=None, limit=50):
    self._pattern = (pattern != r'.*') and re.compile(pattern)
    self._name = name
    self._limit = limit

  @timer.section('terminal')
  def __call__(self, summaries):
    step = max(s for s, _, _, in summaries)
    scalars = {
        k: float(v) for _, k, v in summaries
        if isinstance(v, np.ndarray) and len(v.shape) == 0}
    if self._pattern:
      scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
    else:
      truncated = 0
      if len(scalars) > self._limit:
        truncated = len(scalars) - self._limit
        scalars = dict(list(scalars.items())[:self._limit])
    formatted = {k: self._format_value(v) for k, v in scalars.items()}
    if self._name:
      header = f'{"-" * 20}[{self._name} Step {step}]{"-" * 20}'
    else:
      header = f'{"-" * 20}[Step {step}]{"-" * 20}'
    content = ''
    if self._pattern:
      content += f"Metrics filtered by: '{self._pattern.pattern}'"
    elif truncated:
      content += f'{truncated} metrics truncated, filter to see specific keys.'
    content += '\n'
    if formatted:
      content += ' / '.join(f'{k} {v}' for k, v in formatted.items())
    else:
      content += 'No metrics.'
    printing.print_(f'\n{header}\n{content}\n', flush=True)

  def _format_value(self, value):
    value = float(value)
    if value == 0:
      return '0'
    elif 0.01 < abs(value) < 10000:
      value = f'{value:.2f}'
      value = value.rstrip('0')
      value = value.rstrip('0')
      value = value.rstrip('.')
      return value
    else:
      value = f'{value:.1e}'
      value = value.replace('.0e', 'e')
      value = value.replace('+0', '')
      value = value.replace('+', '')
      value = value.replace('-0', '-')
    return value


class JSONLOutput(AsyncOutput):

  def __init__(
      self, logdir, filename='metrics.jsonl', pattern=r'.*',
      strings=False, parallel=True):
    super().__init__(self._write, parallel)
    self._pattern = re.compile(pattern)
    self._strings = strings
    logdir = path.Path(logdir)
    logdir.mkdir()
    self._filename = logdir / filename

  @timer.section('jsonl')
  def _write(self, summaries):
    bystep = collections.defaultdict(dict)
    for step, name, value in summaries:
      if not self._pattern.search(name):
        continue
      if isinstance(value, str) and self._strings:
        bystep[step][name] = value
      if isinstance(value, np.ndarray) and len(value.shape) == 0:
        bystep[step][name] = float(value)
    lines = ''.join([
        json.dumps({'step': step, **scalars}) + '\n'
        for step, scalars in bystep.items()])
    printing.print_(f'Writing metrics: {self._filename}')
    with self._filename.open('a') as f:
      f.write(lines)


class TensorBoardOutput(AsyncOutput):

  def __init__(
      self, logdir, fps=20, videos=True, maxsize=1e9, parallel=True):
    super().__init__(self._write, parallel)
    self._logdir = str(logdir)
    if self._logdir.startswith('/gcs/'):
      self._logdir = self._logdir.replace('/gcs/', 'gs://')
    self._fps = fps
    self._writer = None
    self._maxsize = self._logdir.startswith('gs://') and maxsize
    self._videos = videos
    if self._maxsize:
      self._checker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
      self._promise = None
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.set_visible_devices([], 'TPU')

  @timer.section('tensorboard_write')
  def _write(self, summaries):
    import tensorflow as tf
    reset = False
    if self._maxsize:
      result = self._promise and self._promise.result()
      # print('Current TensorBoard event file size:', result)
      reset = (self._promise and result >= self._maxsize)
      self._promise = self._checker.submit(self._check)
    if not self._writer or reset:
      print('Creating new TensorBoard event file writer.')
      self._writer = tf.summary.create_file_writer(
          self._logdir, flush_millis=1000, max_queue=10000)
    self._writer.set_as_default()
    for step, name, value in summaries:
      try:
        if isinstance(value, str):
          tf.summary.text(name, value, step)
        elif len(value.shape) == 0:
          tf.summary.scalar(name, value, step)
        elif len(value.shape) == 1:
          if len(value) > 1024:
            value = value.copy()
            np.random.shuffle(value)
            value = value[:1024]
          tf.summary.histogram(name, value, step)
        elif len(value.shape) == 2:
          tf.summary.image(name, value[None, ..., None], step)
        elif len(value.shape) == 3:
          tf.summary.image(name, value[None], step)
        elif len(value.shape) == 4 and self._videos:
          self._video_summary(name, value, step)
      except Exception as e:
        print(f'Error writing summary {name}: {e}')
        raise e
    self._writer.flush()

  @timer.section('tensorboard_check')
  def _check(self):
    import tensorflow as tf
    events = tf.io.gfile.glob(self._logdir.rstrip('/') + '/events.out.*')
    return tf.io.gfile.stat(sorted(events)[-1]).length if events else 0

  @timer.section('tensorboard_video')
  def _video_summary(self, name, video, step):
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
    name = name if isinstance(name, str) else name.decode('utf-8')
    assert video.dtype in (np.float32, np.uint8), (video.shape, video.dtype)
    if np.issubdtype(video.dtype, np.floating):
      video = np.clip(255 * video, 0, 255).astype(np.uint8)
    try:
      T, H, W, C = video.shape
      summary = tf1.Summary()
      image = tf1.Summary.Image(height=H, width=W, colorspace=C)
      image.encoded_image_string = encode_gif(video, self._fps)
      summary.value.add(tag=name, image=image)
      tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
      print('GIF summaries require ffmpeg in $PATH.', e)
      tf.summary.image(name, video, step)


class WandBOutput:

  def __init__(self, name, pattern=r'.*', **kwargs):
    self._pattern = re.compile(pattern)
    import wandb
    wandb.init(name=name, **kwargs)
    self._wandb = wandb

  def __call__(self, summaries):
    bystep = collections.defaultdict(dict)
    wandb = self._wandb
    for step, name, value in summaries:
      if not self._pattern.search(name):
        continue
      if isinstance(value, str):
        bystep[step][name] = value
      elif len(value.shape) == 0:
        bystep[step][name] = float(value)
      elif len(value.shape) == 1:
        bystep[step][name] = wandb.Histogram(value)
      elif len(value.shape) in (2, 3):
        value = value[..., None] if len(value.shape) == 2 else value
        assert value.shape[3] in [1, 3, 4], value.shape
        if value.dtype != np.uint8:
          value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
        value = np.transpose(value, [2, 0, 1])
        bystep[step][name] = wandb.Image(value)
      elif len(value.shape) == 4:
        assert value.shape[3] in [1, 3, 4], value.shape
        value = np.transpose(value, [0, 3, 1, 2])
        if value.dtype != np.uint8:
          value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
        bystep[step][name] = wandb.Video(value)

    for step, metrics in bystep.items():
      self._wandb.log(metrics, step=step)


class ExpaOutput:

  def __init__(self, exp, run, project, user, config=None):
    try:
      import expa
      print(f'Expa: {exp}/{run} ({project})')
      self._expa = expa.Logger(
          exp, run, project, user, api_url='pubsub://expa-dev/ingest')
      if config:
        self._expa.log_params(dict(config))
    except Exception as e:
      print(f'Error exporting Expa: {e}')
      self._expa = None
      return

  def __call__(self, summaries):
    if not self._expa:
      return
    bystep = collections.defaultdict(dict)
    for step, name, value in summaries:
      bystep[step][name] = value
    for step, metrics in bystep.items():
      self._expa.log(metrics, step)
