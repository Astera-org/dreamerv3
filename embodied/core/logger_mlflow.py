import collections
from numbers import Number
from pathlib import Path
import mlflow
import numpy as np

from ._encode_gif import encode_gif


class MLFlowOutput:

  def __init__(self, config, tracking_uri: str | Path, experiment_name: str | None = None):
    assert isinstance(config.logdir, str)
    assert isinstance(config.task, str)
    assert isinstance(config.run.log_video_fps, int)
    self._config = config
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.config.enable_async_logging(True)
    self._run = mlflow.start_run()
    mlflow.log_artifact(Path(config.logdir) / 'config.yaml', 'config.yaml', self._run.info.run_id)

  def __call__(self, summaries):
    bystep = collections.defaultdict(dict)
    for step, name, value in summaries:
      bystep[step][name] = value
    for step, metrics in bystep.items():
      for name, value in metrics.items():
        if isinstance(value, str):
            # TODO: Collect text logs and send at the end? 
            # mlflow.log_text(value)
            pass
        elif isinstance(value, Number):
          mlflow.log_metric(name, float(value), step=step)
        elif isinstance(value, np.ndarray):
          rank = len(value.shape)
          if rank == 0:
            value = value.item()
            if isinstance(value, str):
              # TODO: Collect text logs and send at the end? 
              # mlflow.log_text(value)
              pass
            elif isinstance(value, Number):
              mlflow.log_metric(name, float(value), step=step)
            else:
              # TODO: What is this and how do we log it?
              pass
          elif rank == 1:
            # TODO: Support vectors?
            pass
          elif rank in (2,3):
            mlflow.log_image(value, key=name, step=step)
            pass
          elif rank == 4:
            if np.issubdtype(value.dtype, np.floating):
              value = np.clip(255 * value, 0, 255).astype(np.uint8)
            gif_filename = f"video_step_{step:06}.gif"
            gif_path = Path(self._config.logdir) / gif_filename
            gif = encode_gif(value, self._config.run.log_video_fps)
            with open(gif_path, "wb") as f:
              f.write(gif)
            mlflow.log_artifact(gif_path, "videos")
          else:
            raise ValueError("too many dimensions")

  def __del__(self):
    self._run.__exit__(None, None, None)

