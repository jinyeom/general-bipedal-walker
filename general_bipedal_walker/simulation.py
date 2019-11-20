import math
import numpy as np
import Box2D
from Box2D.b2 import (
  edgeShape, 
  circleShape, 
  fixtureDef, 
  polygonShape, 
  revoluteJointDef, 
  contactListener
)
from .color import Color

class Terrain:
  GRASS = 0
  STUMP = 1
  STAIRS = 2 
  PIT = 3

  @staticmethod
  def rand(np_random=None, include_grass=False):
    if np_random is None:
      np_random = np.random
    options = [Terrain.STUMP, Terrain.STAIRS, Terrain.PIT]
    if include_grass:
      options.append(Terrain.GRASS)
    return np_random.choice(options)
      
class Simulation:
  _FPS                  = 50
  _SCALE                = 30.0
  _VIEWPORT_WIDTH       = 600
  _VIEWPORT_HEIGHT      = 400
  _TERRAIN_STEP         = 14 / _SCALE
  _TERRAIN_LENGTH       = 200
  _TERRAIN_HEIGHT       = _VIEWPORT_HEIGHT / _SCALE / 4
  _TERRAIN_GRASS        = 10 
  _TERRAIN_STARTPAD     = 20
  _FRICTION             = 2.5
  _BIPED_LIMIT          = 1600
  _BIPED_HARDCORE_LIMIT = 2000

  def __init__(self, np_random, hardcore):
    self.np_random = np_random
    self.hardcore = hardcore
    self.world = Box2D.b2World()
    self.terrain = None
    self.fd_polygon = fixtureDef(
      shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]), 
      friction=self._FRICTION
    )
    self.fd_edge = fixtureDef(
      shape=edgeShape(vertices=[(0, 0), (1, 1)]), 
      friction=self._FRICTION, 
      categoryBits=0x0001
    )
    
  @property
  def fps(self):
    return self._FPS

  @property
  def scale(self):
    return self._SCALE

  @property
  def viewport_width(self):
    return self._VIEWPORT_WIDTH

  @property
  def viewport_height(self):
    return self._VIEWPORT_HEIGHT

  @property
  def scaled_width(self):
    return self.viewport_width / self.scale

  @property
  def scaled_height(self):
    return self.viewport_height / self.scale

  @property
  def terrain_step(self):
    return self._TERRAIN_STEP

  @property
  def terrain_length(self):
    return self._TERRAIN_LENGTH

  @property
  def terrain_height(self):
    return self._TERRAIN_HEIGHT

  @property
  def terrain_grass(self):
    return self._TERRAIN_GRASS

  @property
  def terrain_startpad(self):
    return self._TERRAIN_STARTPAD

  @property
  def limit(self):
    return self._BIPED_HARDCORE_LIMIT if self.hardcore else self._BIPED_LIMIT

  def generate_terrain(self):
    state = Terrain.GRASS
    velocity = 0.0
    y = self.terrain_height
    counter = self.terrain_startpad
    oneshot = False

    self.terrain = []
    self.terrain_x = []
    self.terrain_y = []

    for i in range(self.terrain_length):
      x = i * self.terrain_step
      self.terrain_x.append(x)

      if state == Terrain.GRASS and not oneshot:
        sign = np.sign(self.terrain_height - y)
        velocity = 0.8 * velocity + sign * 0.01
        if i > self.terrain_startpad:
          noise = self.np_random.uniform(-1, 1)
          velocity += noise / self.scale
        y += velocity

      elif state == Terrain.PIT and oneshot:
        color1 = Color.rand()
        color2 = Color.BLACK

        counter = self.np_random.randint(3, 5)
        poly = [
          (x, y), 
          (x + self.terrain_step, y), 
          (x + self.terrain_step, y - 4 * self.terrain_step), 
          (x, y - 4 * self.terrain_step)
        ]
        self.fd_polygon.shape.vertices = poly
        t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
        t.color1 = color1
        t.color2 = color2
        self.terrain.append(t)

        self.fd_polygon.shape.vertices = [(x + self.terrain_step * counter, y) for x, y in poly]
        t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
        t.color1 = color1
        t.color2 = color2
        self.terrain.append(t)

        counter += 2
        original_y = y

      elif state == Terrain.PIT and not oneshot:
        y = original_y
        if counter > 1:
          y -= 4 * self.terrain_step

      elif state == Terrain.STUMP and oneshot:
        counter = self.np_random.randint(1, 3)
        poly = [
          (x, y), 
          (x + counter * self.terrain_step, y), 
          (x + counter * self.terrain_step, y + counter * self.terrain_step), 
          (x, y + counter * self.terrain_step),
        ]
        self.fd_polygon.shape.vertices = poly

        t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
        t.color1 = Color.rand()
        t.color2 = Color.BLACK
        self.terrain.append(t)

      elif state == Terrain.STAIRS and oneshot:
        color1 = Color.rand()
        color2 = Color.BLACK

        stair_steps = self.np_random.randint(3, 5)
        stair_width = self.np_random.randint(4, 5)
        stair_height = 1 if self.np_random.rand() > 0.5 else -1

        original_y = y
        for s in range(stair_steps):
          self.fd_polygon.shape.vertices = [
            (x + (s * stair_width) * self.terrain_step, y + (s * stair_height) * self.terrain_step), 
            (x + ((1 + s) * stair_width) * self.terrain_step, y + (s * stair_height) * self.terrain_step),
            (x + ((1 + s) * stair_width) * self.terrain_step, y + (s * stair_height - 1) * self.terrain_step),
            (x + (s * stair_width) * self.terrain_step, y + (s * stair_height - 1) * self.terrain_step)
          ]
          t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
          t.color1 = color1
          t.color2 = color2
          self.terrain.append(t)
        counter = stair_steps * stair_width

      elif state == Terrain.STAIRS and not oneshot:
        s = stair_steps * stair_width - counter - stair_height
        n = s / stair_width
        y = original_y + (n * stair_height) * self.terrain_step

      self.terrain_y.append(y)
      oneshot = False
      counter -= 1

      if counter == 0:
        counter = self.np_random.randint(self.terrain_grass / 2, self.terrain_grass)
        
        if state == Terrain.GRASS and self.hardcore:
          state = Terrain.rand(np_random=self.np_random)
        else:
          state = Terrain.GRASS
        oneshot = True

    self.terrain_poly = []
    for i in range(self.terrain_length - 1):
      poly = [
        (self.terrain_x[i], self.terrain_y[i]), 
        (self.terrain_x[i+1], self.terrain_y[i+1])
      ]
      self.fd_edge.shape.vertices = poly
      t = self.world.CreateStaticBody(fixtures=self.fd_edge)
      t.color1 = Color.WHITE if i % 2 == 0 else Color.BLACK
      t.color2 = Color.WHITE if i % 2 == 0 else Color.BLACK
      self.terrain.append(t)

      poly += [(poly[1][0], 0), (poly[0][0], 0)]
      self.terrain_poly.append((poly, Color.DARK_GREEN))
    self.terrain.reverse()
    
  def destroy(self):
    if not self.terrain:
      return
    self.world.contactListener = None
    for t in self.terrain:
      self.world.DestroyBody(t)
    self.terrain = []

  def step(self):
    self.world.Step(1.0 / self.fps, 6 * 30, 2 * 30)
