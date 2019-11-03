import sys

def worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
  assert shared_memory is None
  env = env_fn()
  parent_pipe.close()
  try:
    while True:
      command, data = pipe.recv()
      if command == 'sample':
        env.sample(data)
      elif command == 'reset':
        observation = env.reset()
        pipe.send((observation, True))
      elif command == 'step':
        observation, reward, done, info = env.step(data)
        if done:
          observation = env.reset()
        pipe.send(((observation, reward, done, info), True))
      elif command == 'seed':
        env.seed(data)
        pipe.send((None, True))
      elif command == 'close':
        pipe.send((None, True))
        break
      elif command == '_check_observation_space':
        pipe.send((data == env.observation_space, True))
      else:
        raise RuntimeError(
          'Received unknown command `{0}`. Must '
          'be one of {`reset`, `step`, `seed`, `close`, '
          '`_check_observation_space`}.'.format(command)
        )
  except (KeyboardInterrupt, Exception):
    error_queue.put((index,) + sys.exc_info()[:2])
    pipe.send((None, False))
  finally:
    env.close()