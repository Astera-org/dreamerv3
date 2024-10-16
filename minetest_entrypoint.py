import os
import warnings
from functools import partial as bind

import gymnasium.wrappers.time_limit
import jax
import gymnasium
import dreamerv3
import embodied
import mlflow

warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

def main():
  mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
  mlflow.set_experiment("/Shared/dreamerv3_hafner_minetest")
  mlflow.config.enable_async_logging(True)

  jax.config.update("jax_compilation_cache_dir", "./jax_cache/")
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

  config = embodied.Config(dreamerv3.Agent.configs['defaults'])
  config = config.update({
      **dreamerv3.Agent.configs['size50m'],
      'logdir': f'./logs/{embodied.timestamp()}/',
      'run.log_video_fps': 6,
      'run.train_ratio': 512,
      'run.eval_every': 600, # seconds
      # Set 'run.driver_parallel': False and 'run.num_envs': 1 to see stacktrace when env fails.
      'run.num_envs': 4,
      'run.num_envs_eval': 1,
      'enc.simple.minres': 8, # solves shape mismatch
      'dec.simple.minres': 8, # solves shape mismatch
      'enc.spaces': 'image|health|hunger|thirst',
      'dec.spaces': 'image|health|hunger|thirst',
      'batch_size': 2, # oom
  })
  config = embodied.Flags(config).parse()

  print('Logdir:', config.logdir)
  logdir = embodied.Path(config.logdir)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')

  def make_agent(config):
    env = make_env(config)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    env.close()
    return agent

  def make_logger(config):
    logdir = embodied.Path(config.logdir)
    return embodied.Logger(embodied.Counter(), [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        embodied.logger.MLFlowOutput()
        # embodied.logger.WandbOutput(logdir.name, config=config),
    ])

  def make_replay(config, directory=None, is_eval=False, rate_limit=False):
    directory = directory and embodied.Path(config.logdir) / directory
    size = int(config.replay.size / 10 if is_eval else config.replay.size)
    length = config.replay_length_eval or config.batch_length_eval if is_eval else config.replay_length or config.batch_length
    kwargs = {}
    kwargs['online'] = config.replay.online
    if rate_limit and config.run.train_ratio > 0:
      kwargs['samples_per_insert'] = config.run.train_ratio / (
          length - config.replay_context)
      kwargs['tolerance'] = 5 * config.batch_size
      kwargs['min_size'] = min(
          max(config.batch_size, config.run.train_fill), size)
    selectors = embodied.replay.selectors
    if config.replay.fracs.uniform < 1 and not is_eval:
      assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
          'Gradient scaling for low-precision training can produce invalid loss '
          'outputs that are incompatible with prioritized replay.')
      import numpy as np
      recency = 1.0 / np.arange(1, size + 1) ** config.replay.recexp
      kwargs['selector'] = selectors.Mixture(dict(
          uniform=selectors.Uniform(),
          priority=selectors.Prioritized(**config.replay.prio),
          recency=selectors.Recency(recency),
      ), config.replay.fracs)
    kwargs['chunksize'] = config.replay.chunksize
    replay = embodied.replay.Replay(length, size, directory, **kwargs)
    return replay

  def make_env(config, env_id=0):
    from embodied.envs import from_gymnasium
    from embodied.envs.minetest_wrapper import MinetestWrapper
    env = MinetestWrapper("boad")
    # env = gymnasium.wrappers.TimeLimit(env, 1000)
    # env = gymnasium.wrappers.RecordVideo(env, config.logdir + "/video", lambda _: True, lambda _: True)
    return dreamerv3.wrap_env(from_gymnasium.FromGymnasium(env), config)

  def make_eval_env(config, env_id=0):
    from embodied.envs import from_gymnasium
    from embodied.envs.minetest_wrapper import MinetestWrapper
    env = MinetestWrapper("boad")
    env = gymnasium.wrappers.TimeLimit(env, 1000)
    env = gymnasium.wrappers.RecordVideo(env, config.logdir + f"/video-{env_id:02}", episode_trigger=lambda _: True)
    return dreamerv3.wrap_env(from_gymnasium.FromGymnasium(env), config)

  args = embodied.Config(
      **config.run,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      batch_length_eval=config.batch_length_eval,
      replay_context=config.replay_context,
  )

  with mlflow.start_run():
    mlflow.log_artifact(logdir / 'config.yaml')

    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', is_eval=True),
        bind(make_env, config),
        bind(make_eval_env, config),
        bind(make_logger, config), args)


if __name__ == '__main__':
  main()
