import math
import numpy as np
import gym
from gym import spaces
from gym.utils import colorize, seeding
from gym.envs.classic_control import rendering
import Box2D
from Box2D.b2 import (
  edgeShape, 
  circleShape, 
  fixtureDef, 
  polygonShape, 
  revoluteJointDef, 
  contactListener
)
from .simulation import Simulation
from .robot import RobotConfig, BipedalRobot
from .color import Color

class ContactDetector(contactListener):
  def __init__(self, env):
    contactListener.__init__(self)
    self.env = env

  def BeginContact(self, contact):
    if (self.env.robot.hull.body == contact.fixtureA.body or 
        self.env.robot.hull.body == contact.fixtureB.body):
      self.env.game_over = True
    for leg in [self.env.robot.leg1.bot_body, self.env.robot.leg2.bot_body]:
      if leg in [contact.fixtureA.body, contact.fixtureB.body]:
        leg.ground_contact = True

  def EndContact(self, contact):
    for leg in [self.env.robot.leg1.bot_body, self.env.robot.leg2.bot_body]:
      if leg in [contact.fixtureA.body, contact.fixtureB.body]:
        leg.ground_contact = False

class GeneralBipedalWalker(gym.Env):
  def __init__(self, hardcore=False):
    self.hardcore = hardcore
    self.viewer = None
    
    self.seed()
    self.sim = Simulation(self.np_random, hardcore)
    self.robot = BipedalRobot(self.sim, RobotConfig())

    os_lim = np.array([np.inf for _ in range(24)])
    as_lim = np.array([1 for _ in range(4)])
    self.observation_space = spaces.Box(-os_lim, os_lim)
    self.action_space = spaces.Box(-as_lim, as_lim)

    self.reset()

  def sample(self, symmetric=True):
    self.robot.destroy()
    config = RobotConfig.sample(self.np_random, symmetric=symmetric)
    self.robot = BipedalRobot(self.sim, config)

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _destroy(self):
    self.robot.destroy()
    self.sim.destroy()

  def reset(self):
    self._destroy()

    self.sim.world.contactListener_bug_workaround = ContactDetector(self)
    self.sim.world.contactListener = self.sim.world.contactListener_bug_workaround

    init_x = self.sim.terrain_step * self.sim.terrain_startpad / 2
    init_y = self.sim.terrain_height + np.maximum(
      self.robot.leg1.top_height + self.robot.leg1.bot_height, 
      self.robot.leg2.top_height + self.robot.leg2.bot_height
    )
    init_noise = (self.np_random.uniform(-5, 5), 0)

    self.sim.generate_terrain()
    self.robot.reset(init_x, init_y, init_noise)
    self.assets = self.sim.terrain + self.robot.parts
    
    self.game_over = False
    self.prev_shaping = None
    self.scroll = 0.0
    self.lidar_render = 0
    self.timer = 0

    return self.step(np.array([0, 0, 0, 0]))[0]

  def reward(self, state, action):
    if self.game_over or self.robot.hull.body.position[0] < 0:
      return -100
    reward = 0
    shaping = 130 * self.robot.hull.body.position.x / self.sim.scale
    shaping -= 5.0 * abs(state[0])
    if self.prev_shaping is not None:
      reward = shaping - self.prev_shaping
    self.prev_shaping = shaping
    for a in action:
      reward -= (
        0.00035 * 
        self.robot.config.motors_torque * 
        np.clip(np.abs(a), 0, 1)
      )
    return reward

  def done(self):
    done = False
    if self.game_over or self.robot.hull.body.position[0] < 0:
      done = True
    goal = (
      self.sim.terrain_step *
      (self.sim.terrain_length - self.sim.terrain_grass)
    )
    if self.robot.hull.body.position[0] > goal:
      done = True
    if self.timer >= self.sim.limit:
      done = True
    return done

  def step(self, action):
    joints = self.robot.step(action)
    self.sim.step()
    pos = self.robot.hull.body.position
    vel = self.robot.hull.body.linearVelocity

    # Update the environment state.
    joint_state = [
      self.robot.hull.body.angle,
      2.0 * self.robot.hull.body.angularVelocity / self.sim.fps,
      0.3 * vel.x * self.sim.scaled_width / self.sim.fps,
      0.3 * vel.y * self.sim.scaled_height / self.sim.fps,
      joints[0].angle,
      joints[0].speed / self.robot.config.speed_hip,
      joints[1].angle + 1.0,
      joints[1].speed / self.robot.config.speed_knee,
      1.0 if self.robot.leg1.bot_body.ground_contact else 0.0,
      joints[2].angle,
      joints[2].speed / self.robot.config.speed_hip,
      joints[3].angle + 1.0,
      joints[3].speed / self.robot.config.speed_knee,
      1.0 if self.robot.leg2.bot_body.ground_contact else 0.0
    ]
    lidar_state = [lidar.fraction for lidar in self.robot.scan(pos)]
    state = np.array(joint_state + lidar_state, dtype=np.float32)
    self.scroll = pos[0] - self.sim.scaled_width / 5

    # Compute reward.
    reward = self.reward(state, action)

    # Determine whether the environment is in terminal state.
    done = self.done()

    self.timer += 1
    return state, reward, done, {}

  def render_sky(self):
    self.viewer.draw_polygon([
        (                        self.scroll,                      0), 
        (self.scroll + self.sim.scaled_width,                      0),
        (self.scroll + self.sim.scaled_width, self.sim.scaled_height), 
        (                        self.scroll, self.sim.scaled_height)
      ], 
      color=Color.LIGHT_GRAY
    )

  def render_terrain(self):
    for poly, color in self.sim.terrain_poly:
      if poly[1][0] < self.scroll:
        continue
      if poly[0][0] > self.scroll + self.sim.scaled_width:
        continue
      self.viewer.draw_polygon(poly, color=color)

  def render_lidar(self):
    self.lidar_render = (self.lidar_render + 1) % 100
    idx = self.lidar_render
    if idx < 2 * len(self.robot.lidar.callbacks):
      if idx < len(self.robot.lidar.callbacks):
        line = self.robot.lidar.callbacks[idx]
      else:
        idx = len(self.robot.lidar.callbacks) - idx - 1
        line = self.robot.lidar.callbacks[idx]
      self.viewer.draw_polyline(
        [line.p1, line.p2],
        color=Color.RED, 
        linewidth=2
      )

  def render_assets(self):
    for obj in self.assets:
      for f in obj.fixtures:
        trans = f.body.transform
        if type(f.shape) is circleShape:
          t = rendering.Transform(translation=trans*f.shape.pos)
          self.viewer.draw_circle(
            f.shape.radius, 
            30, 
            color=obj.color1
          ).add_attr(t)
          self.viewer.draw_circle(
            f.shape.radius, 
            30, 
            color=obj.color2, 
            filled=False, 
            linewidth=2
          ).add_attr(t)
        else:
          path = [trans * v for v in f.shape.vertices]
          self.viewer.draw_polygon(path, color=obj.color1)
          path.append(path[0])
          self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

  def render_flags(self):
    flagy1 = self.sim.terrain_height
    flagy2 = flagy1 + 50 / self.sim.scale
    x = self.sim.terrain_step * 3
    self.viewer.draw_polyline([
        (x, flagy1), 
        (x, flagy2)
      ], 
      color=Color.BLACK,
      linewidth=2
    )
    f = [
      (                      x,                       flagy2), 
      (                      x, flagy2 - 10 / self.sim.scale), 
      (x + 25 / self.sim.scale,  flagy2 - 5 / self.sim.scale)
    ]
    self.viewer.draw_polygon(f, color=Color.RED)
    self.viewer.draw_polyline(f + [f[0]], color=Color.BLACK, linewidth=2)

  def render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    if self.viewer is None:
      self.viewer = rendering.Viewer(
        self.sim.viewport_width, 
        self.sim.viewport_height
      )

    self.viewer.set_bounds(
      self.scroll, 
      self.sim.scaled_width + self.scroll, 
      0, 
      self.sim.scaled_height
    )

    self.render_sky()
    self.render_terrain()
    self.render_lidar()
    self.render_assets()
    self.render_flags()

    return self.viewer.render(return_rgb_array=mode=='rgb_array')
