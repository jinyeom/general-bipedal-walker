import sys
from copy import deepcopy
from enum import Enum
from gym.vector import AsyncVectorEnv as AsyncVectorEnv_
from gym.error import AlreadyPendingCallError
from gym.vector.utils import write_to_shared_memory, concatenate

class AsyncVectorEnv(AsyncVectorEnv_):
  def sample(self, symmetric=True):
    self._assert_is_running()
    for pipe in self.parent_pipes:
      pipe.send(('sample', symmetric))
    _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
    self._raise_if_errors(successes)
    return deepcopy(self.observations) if self.copy else self.observations

def worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
  assert shared_memory is not None
  env = env_fn()
  observation_space = env.observation_space
  parent_pipe.close()
  try:
    while True:
      command, data = pipe.recv()
      if command == 'sample':
        observation = env.sample(data)
        write_to_shared_memory(index, observation, shared_memory, observation_space)
        pipe.send((None, True))
      elif command == 'reset':
        observation = env.reset()
        write_to_shared_memory(index, observation, shared_memory, observation_space)
        pipe.send((None, True))
      elif command == 'step':
        observation, reward, done, info = env.step(data)
        if done:
          observation = env.reset()
        write_to_shared_memory(index, observation, shared_memory, observation_space)
        pipe.send(((None, reward, done, info), True))
      elif command == 'seed':
        env.seed(data)
        pipe.send((None, True))
      elif command == 'close':
        pipe.send((None, True))
        break
      elif command == '_check_observation_space':
        pipe.send((data == observation_space, True))
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