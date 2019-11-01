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
from simulation import Simulation
from robot import RobotConfig, BipedalRobot
from color import Color

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

  def augment(self, params):
    self.robot.destroy()
    self.robot = BipedalRobot(self.sim, RobotConfig(params))

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

    self.sim.reset()
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
    if (self.robot.hull.body.position[0] > (
        (self.sim.terrain_length - self.sim.terrain_grass) * 
        self.sim.terrain_step)):
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
      color=Color.lighter(Color.LIGHT_GRAY)
    )

  def render_cloud(self):
    for poly, x1, x2 in self.sim.cloud_poly:
      if x2 < self.scroll / 2:
        continue
      if x1 > self.scroll / 2 + self.sim.scaled_width:
        continue
      self.viewer.draw_polygon([
          (p[0] + self.scroll / 2, p[1]) 
          for p in poly
        ], 
        color=Color.WHITE
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
    i = self.lidar_render
    if i < 2 * len(self.robot.lidar.callbacks):
      if i < len(self.robot.lidar.callbacks):
        l = self.robot.lidar.callbacks[i]
      else:
        idx = len(self.robot.lidar.callbacks) - i - 1
        l = self.robot.lidar.callbacks[idx]
      self.viewer.draw_polyline(
        [l.p1, l.p2], 
        color=Color.RED, 
        linewidth=1
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
    self.render_cloud()
    self.render_terrain()
    self.render_lidar()
    self.render_assets()
    self.render_flags()

    return self.viewer.render(return_rgb_array=mode=='rgb_array')

def demo():
  # Heurisic: suboptimal, have no notion of balance.
  env = GeneralBipedalWalker(hardcore=False)
  # env.sample()
  env.reset()
  steps = 0
  total_reward = 0
  a = np.array([0.0, 0.0, 0.0, 0.0])
  STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
  SPEED = 0.29  # Will fall forward on higher speed
  state = STAY_ON_ONE_LEG
  moving_leg = 0
  supporting_leg = 1 - moving_leg
  SUPPORT_KNEE_ANGLE = +0.1
  supporting_knee_angle = SUPPORT_KNEE_ANGLE
  while True:
    env.render(mode='human')

    s, r, done, info = env.step(a)
    total_reward += r
    if steps % 20 == 0 or done:
      print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
      print("step {} total_reward {:+0.2f}".format(steps, total_reward))
      print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4]]))
      print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9]]))
      print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
    steps += 1

    contact0 = s[8]
    contact1 = s[13]
    moving_s_base = 4 + 5*moving_leg
    supporting_s_base = 4 + 5*supporting_leg

    hip_targ  = [None, None] # -0.8 .. +1.1
    knee_targ = [None, None] # -0.6 .. +0.9
    hip_todo  = [0.0, 0.0]
    knee_todo = [0.0, 0.0]

    if state == STAY_ON_ONE_LEG:
      hip_targ[moving_leg] = 1.1
      knee_targ[moving_leg] = -0.6
      supporting_knee_angle += 0.03
      if s[2] > SPEED:
        supporting_knee_angle += 0.03
      supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
      knee_targ[supporting_leg] = supporting_knee_angle
      if s[supporting_s_base+0] < 0.10: # supporting leg is behind
        state = PUT_OTHER_DOWN

    if state == PUT_OTHER_DOWN:
      hip_targ[moving_leg] = 0.1
      knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
      knee_targ[supporting_leg] = supporting_knee_angle
      if s[moving_s_base+4]:
        state = PUSH_OFF
        supporting_knee_angle = min(s[moving_s_base+2], SUPPORT_KNEE_ANGLE)

    if state == PUSH_OFF:
      knee_targ[moving_leg] = supporting_knee_angle
      knee_targ[supporting_leg] = 1.0
      if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
        state = STAY_ON_ONE_LEG
        moving_leg = 1 - moving_leg
        supporting_leg = 1 - moving_leg

    if hip_targ[0]:
      hip_todo[0] = 0.9*(hip_targ[0]-s[4]) - 0.25*s[5]
    if hip_targ[1]:
      hip_todo[1] = 0.9*(hip_targ[1]-s[9]) - 0.25*s[10]
    if knee_targ[0]:
      knee_todo[0] = 4.0*(knee_targ[0]-s[6])  - 0.25*s[7]
    if knee_targ[1]:
      knee_todo[1] = 4.0*(knee_targ[1]-s[11]) - 0.25*s[12]

    hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
    hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
    knee_todo[0] -= 15.0 * s[3] # vertical speed, to damp oscillations
    knee_todo[1] -= 15.0 * s[3]

    a[0] = hip_todo[0]
    a[1] = knee_todo[0]
    a[2] = hip_todo[1]
    a[3] = knee_todo[1]
    a = np.clip(0.5*a, -1.0, 1.0)

    env.render()
    if done:
      break

if __name__ == '__main__':
  demo()
