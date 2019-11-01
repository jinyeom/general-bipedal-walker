import numpy as np

class Color:
  BLACK       = np.array((0.000, 0.000, 0.000))
  DARK_BLUE   = np.array((0.114, 0.169, 0.325))
  DARK_PURPLE = np.array((0.494, 0.145, 0.325))
  DARK_GREEN  = np.array((0.000, 0.529, 0.318))
  BROWN       = np.array((0.671, 0.322, 0.212))
  DARK_GRAY   = np.array((0.373, 0.341, 0.310))
  LIGHT_GRAY  = np.array((0.761, 0.765, 0.780))
  WHITE       = np.array((1.000, 0.945, 0.910))
  RED         = np.array((1.000, 0.000, 0.302))
  ORANGE      = np.array((1.000, 0.639, 0.000))
  YELLOW      = np.array((1.000, 0.925, 0.153))
  GREEN       = np.array((0.000, 0.894, 0.212))
  BLUE        = np.array((0.161, 0.678, 1.000))
  INDIGO      = np.array((0.514, 0.463, 0.612))
  PINK        = np.array((1.000, 0.467, 0.659))
  PEACH       = np.array((1.000, 0.800, 0.667))

  @staticmethod
  def darker(color, scale=1):
    return np.clip(color - scale * 0.1, 0, 1)

  @staticmethod
  def lighter(color, scale=1):
    return np.clip(color + scale * 0.1, 0, 1)

  @staticmethod
  def rand(rng=None, include_grays=False):
    colors = [
      'DARK_BLUE',
      'DARK_PURPLE',
      'DARK_GREEN',
      'BROWN',
      'RED',
      'ORANGE',
      'YELLOW',
      'GREEN',
      'BLUE',
      'INDIGO',
      'PINK',
      'PEACH',
    ]
    if include_grays:
      colors += [
        'BLACK',
        'DARK_GRAY',
        'LIGHT_GRAY', 
        'WHITE',
      ]
    if rng:
      return Color.__dict__[rng.choice(colors)]
    return Color.__dict__[np.random.choice(colors)]