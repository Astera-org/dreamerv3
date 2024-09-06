import os
import warnings
from functools import partial as bind

import jax

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
      # Set 'run.driver_parallel': False and 'run.num_envs': 1 to see stacktrace when env fails.
      'run.num_envs': 32,
      'enc.simple.minres': 8, # solves shape mismatch
      'dec.simple.minres': 8, # solves shape mismatch
      'enc.spaces': 'image|health|hunger|thirst',
      'dec.spaces': 'image|health|hunger|thirst',
      'batch_size': 1, # oom
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

  def make_replay(config):
    return embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=embodied.Path(config.logdir) / 'replay',
        online=config.replay.online)

  def make_env(config, env_id=0):
    from embodied.envs import from_gymnasium
    from embodied.envs.minetest_wrapper import MinetestWrapper
    return dreamerv3.wrap_env(from_gymnasium.FromGymnasium(MinetestWrapper("boad")), config)

  args = embodied.Config(
      **config.run,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      batch_length_eval=config.batch_length_eval,
      replay_context=config.replay_context,
  )

  with mlflow.start_run():
    mlflow.log_artifact(logdir / 'config.yaml', 'config.yaml')

    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config),
        bind(make_env, config),
        bind(make_logger, config), args)


if __name__ == '__main__':
  main()
