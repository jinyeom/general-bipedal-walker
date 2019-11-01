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
from color import Color

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
      shape=polygonShape(
        vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]
      ), 
      friction=self._FRICTION
    )
    self.fd_edge = fixtureDef(
      shape=edgeShape(
        vertices=[(0, 0), (1, 1)]
      ), 
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

  def _pit_poly(self, x, y, counter):
    return [
      (                   x,                      y), 
      (x+self._TERRAIN_STEP,                      y), 
      (x+self._TERRAIN_STEP, y-4*self._TERRAIN_STEP), 
      (                   x, y-4*self._TERRAIN_STEP)
    ]

  def _stump_poly(self, x, y, counter):
    return [
      (                           x,                            y), 
      (x+counter*self._TERRAIN_STEP,                            y), 
      (x+counter*self._TERRAIN_STEP, y+counter*self._TERRAIN_STEP), 
      (                           x, y+counter*self._TERRAIN_STEP),
    ]

  def _stair_poly(self, s, x, y, w, h):
    return [
      (    x+(s*w)*self._TERRAIN_STEP,   y+(s*h)*self._TERRAIN_STEP), 
      (x+((1+s)*w)*self._TERRAIN_STEP,   y+(s*h)*self._TERRAIN_STEP),
      (x+((1+s)*w)*self._TERRAIN_STEP, y+(s*h-1)*self._TERRAIN_STEP),
      (    x+(s*w)*self._TERRAIN_STEP, y+(s*h-1)*self._TERRAIN_STEP)
    ]

  def _generate_terrain(self):
    GRASS, STUMP, STAIRS, PIT = range(4)

    state    = GRASS
    velocity = 0.0
    y        = self._TERRAIN_HEIGHT
    counter  = self._TERRAIN_STARTPAD
    oneshot  = False

    self.terrain   = []
    self.terrain_x = []
    self.terrain_y = []

    for i in range(self._TERRAIN_LENGTH):
      x = i * self._TERRAIN_STEP
      self.terrain_x.append(x)

      if state == GRASS and not oneshot:
        sign = np.sign(self._TERRAIN_HEIGHT - y)
        velocity = 0.8 * velocity + sign * 0.01
        if i > self._TERRAIN_STARTPAD:
          noise = self.np_random.uniform(-1, 1)
          velocity += noise / self._SCALE
        y += velocity

      elif state == PIT and oneshot:
        color2 = Color.rand()
        color1 = Color.lighter(color2)

        counter = self.np_random.randint(3, 5)
        poly = self._pit_poly(x, y, counter)
        self.fd_polygon.shape.vertices = poly
        t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
        t.color1 = color1
        t.color2 = color2
        self.terrain.append(t)

        self.fd_polygon.shape.vertices = [
          (x + self._TERRAIN_STEP * counter, y) 
          for x, y in poly
        ]
        t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
        t.color1 = color1
        t.color2 = color2
        self.terrain.append(t)

        counter += 2
        original_y = y

      elif state == PIT and not oneshot:
        y = original_y
        if counter > 1:
          y -= 4 * self._TERRAIN_STEP

      elif state == STUMP and oneshot:
        color2 = Color.rand()
        color1 = Color.lighter(color2)

        counter = self.np_random.randint(1, 3)
        poly = self._stump_poly(x, y, counter)
        self.fd_polygon.shape.vertices = poly
        t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
        t.color1 = color1
        t.color2 = color2
        self.terrain.append(t)

      elif state == STAIRS and oneshot:
        color2 = Color.rand()
        color1 = Color.lighter(color2)

        stair_steps = self.np_random.randint(3, 5)
        stair_width = self.np_random.randint(4, 5)
        stair_height = 1 if self.np_random.rand() > 0.5 else -1
        original_y = y
        for s in range(stair_steps):
          poly = self._stair_poly(s, x, y, stair_width, stair_height)
          self.fd_polygon.shape.vertices = poly
          t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
          t.color1 = color1
          t.color2 = color2
          self.terrain.append(t)
        counter = stair_steps * stair_width

      elif state == STAIRS and not oneshot:
        s = stair_steps * stair_width - counter - stair_height
        n = s / stair_width
        y = original_y + (n * stair_height) * self.terrain_step

      oneshot = False
      self.terrain_y.append(y)
      counter -= 1
      if counter == 0:
        counter = self.np_random.randint(
          self.terrain_grass / 2, 
          self.terrain_grass
        )
        if state == GRASS and self.hardcore:
          state = self.np_random.randint(1, 4)
        else:
          state = GRASS
        oneshot = True

    self.terrain_poly = []
    for i in range(self.terrain_length - 1):
      poly = [
        (  self.terrain_x[i],   self.terrain_y[i]), 
        (self.terrain_x[i+1], self.terrain_y[i+1])
      ]
      self.fd_edge.shape.vertices = poly
      t = self.world.CreateStaticBody(fixtures=self.fd_edge)
      t.color1 = Color.WHITE if i % 2 == 0 else Color.DARK_GRAY
      t.color2 = Color.WHITE if i % 2 == 0 else Color.DARK_GRAY
      self.terrain.append(t)

      poly += [(poly[1][0], 0), (poly[0][0], 0)]
      self.terrain_poly.append((poly, Color.lighter(Color.DARK_GREEN)))
    self.terrain.reverse()

  def _generate_clouds(self):
    self.cloud_poly = []
    for i in range(self.terrain_length // 20):
      x = self.np_random.uniform(0, self.terrain_length) * self.terrain_step
      y = self.viewport_height / self._SCALE * 3 / 4
      poly = [
        (
          (x+15*self.terrain_step*math.sin(math.pi*2*a/5)+
           self.np_random.uniform(0,5*self.terrain_step)),
          (y+5*self.terrain_step*math.cos(math.pi*2*a/5)+
           self.np_random.uniform(0,5*self.terrain_step))
        )
        for a in range(5)
      ]
      x1 = min([p[0] for p in poly])
      x2 = max([p[0] for p in poly])
      self.cloud_poly.append((poly, x1, x2))

  def destroy(self):
    if not self.terrain:
      return
    self.world.contactListener = None
    for t in self.terrain:
      self.world.DestroyBody(t)
    self.terrain = []

  def reset(self):
    self._generate_terrain()
    self._generate_clouds()

  def step(self):
    self.world.Step(1.0 / self.fps, 6 * 30, 2 * 30)
