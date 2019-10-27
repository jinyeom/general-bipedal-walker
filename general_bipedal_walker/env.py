import math
import numpy as np
import gym
from gym import spaces
from gym.utils import colorize, seeding
import Box2D
from Box2D.b2 import (
  edgeShape, 
  circleShape, 
  fixtureDef, 
  polygonShape, 
  revoluteJointDef, 
  contactListener
)
from world import World
from robot import RobotConfig, BipedalRobot

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
    self.world = World(self.np_random, hardcore)
    self.robot = BipedalRobot(self.world, RobotConfig())

    os_lim = np.array([np.inf for _ in range(24)])
    as_lim = np.array([1 for _ in range(4)])
    self.observation_space = spaces.Box(-os_lim, os_lim)
    self.action_space = spaces.Box(-as_lim, as_lim)

    self.reset()

  def augment(self, params):
    self.robot.destroy()
    self.robot = BipedalRobot(self.world, RobotConfig(params))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def destroy(self):
    self.robot.destroy()
    self.world.destroy()

  def reset(self):
    self.destroy()

    self.world.contactListener_bug_workaround = ContactDetector(self)
    self.world.contactListener = self.world.contactListener_bug_workaround

    init_x = self.world.terrain_step * self.world.terrain_startpad / 2
    init_y = self.world.terrain_height + np.maximum(
      self.robot.config.leg1_top_height + self.robot.config.leg1_bot_height, 
      self.robot.config.leg2_top_height + self.robot.config.leg2_bot_height
    )
    init_noise = (self.np_random.uniform(-5, 5), 0)

    self.world.reset()
    self.robot.reset(init_x, init_y, init_noise)
    self.assets = self.world.terrain + self.robot.parts
    
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
    shaping = 130 * self.robot.hull.body.position.x / self.world.scale
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
    if (self.robot.hull.body.position[0] > (
        (self.world.terrain_length - self.world.terrain_grass) * 
        self.world.terrain_step)):
      done = True
    if self.timer >= self.world.limit:
      done = True
    return done

  def step(self, action):
    joints = self.robot.step(action)
    self.world.step()
    lidars = self.robot.scan()

    # Update the environment state.
    joint_state = [
      self.robot.hull.body.angle,
      2.0 * self.robot.hull.body.angularVelocity / self.world.fps,
      (
        0.3 * self.robot.hull.body.linearVelocity.x * 
        (self.world.viewport_width / self.world.scale) / self.world.fps
      ),
      (
        0.3 * self.robot.hull.body.linearVelocity.y * 
        (self.world.viewport_height / self.world.scale) / self.world.fps
      ),
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
    lidar_state = [lidar.fraction for lidar in lidars]
    state = np.array(joint_state + lidar_state, dtype=np.float32)

    self.scroll = (
      self.robot.hull.body.position[0] - 
      self.world.viewport_width / self.world.scale / 5
    )

    # Compute reward.
    reward = self.reward(state, action)

    # Determine whether the environment is in terminal state.
    done = self.done()

    self.timer += 1
    return state, reward, done, {}

  