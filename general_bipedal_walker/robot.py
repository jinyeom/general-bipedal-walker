import math
import numpy as np
from Box2D.b2 import (
  edgeShape, 
  circleShape, 
  fixtureDef, 
  polygonShape, 
  revoluteJointDef, 
  contactListener,
  rayCastCallback
)
from .color import Color

class Hull:
  VERTICES = [(-30,  9), (6, 9), (34, 1), (34, -8), (-30, -8)]

  def __init__(self, config):
    self.color = config.hull_color
    self.body = None
    vertices = [(x / config.scale, y / config.scale) for x, y in Hull.VERTICES]
    self.fixture = fixtureDef(
      shape=polygonShape(vertices=vertices),
      density=5.0,
      friction=0.1,
      categoryBits=0x0020,
      maskBits=0x001, # collide only with ground
      restitution=0.0 # 0.99 bouncy
    )

  @property
  def parts(self):
    return [self.body]

  def reset(self, world, init_x, init_y, noise):
    self.body = world.CreateDynamicBody(
      position=(init_x, init_y), 
      fixtures=self.fixture
    )
    self.body.color1 = self.color
    self.body.color2 = Color.BLACK
    self.body.ApplyForceToCenter(noise, True)

class Lidar:
  class Callback(rayCastCallback):
    def ReportFixture(self, fixture, point, normal, fraction):
      if (fixture.filterData.categoryBits & 1) == 0:
        return 1
      self.p2 = point
      self.fraction = fraction
      return 0

  def __init__(self, config):
    self.scan_range = config.lidar_range
    self.callbacks = None

  def reset(self):
    self.callbacks = [Lidar.Callback() for _ in range(10)]

  def scan(self, world, pos):
    for i, lidar in enumerate(self.callbacks):
      self.callbacks[i].fraction = 1.0
      self.callbacks[i].p1 = pos
      self.callbacks[i].p2 = (
        pos[0] + math.sin(1.5 * i / 10.0) * self.scan_range,
        pos[1] - math.cos(1.5 * i / 10.0) * self.scan_range
      )
      world.RayCast(lidar, lidar.p1, lidar.p2)

class Leg:
  def __init__(self, config, left=True):
    self.left = left
    if self.left:
      self.color = Color.lighter(config.leg_color)
      self.top_width = config.leg1_top_width / config.scale
      self.top_height = config.leg1_top_height / config.scale
      self.bot_width = config.leg1_bot_width / config.scale
      self.bot_height = config.leg1_bot_height / config.scale
    else:
      self.color = Color.darker(config.leg_color)
      self.top_width = config.leg2_top_width / config.scale
      self.top_height = config.leg2_top_height / config.scale
      self.bot_width = config.leg2_bot_width / config.scale
      self.bot_height = config.leg2_bot_height / config.scale
    self.leg_down = -8 / config.scale

    # configure the top leg part 
    self.top_shift = self.top_height / 2 + self.leg_down
    self.top_body = None
    self.top_fixture = fixtureDef(
      shape=polygonShape(box=(self.top_width / 2, self.top_height / 2)),
      density=1.0,
      restitution=0.0,
      categoryBits=0x0020,
      maskBits=0x001
    )

    # configure the motor torque
    self.motors_torque = config.motors_torque
    self.joint = None

    # configure the bottom leg par
    self.bot_shift = self.top_height + self.bot_height / 2 + self.leg_down
    self.bot_body = None
    self.bot_fixture = fixtureDef(
      shape=polygonShape(box=(self.bot_width / 2, self.bot_height / 2)),
      density=1.0,
      restitution=0.0,
      categoryBits=0x0020,
      maskBits=0x001
    )

  @property
  def parts(self):
    return [self.bot_body, self.top_body]

  def reset(self, world, init_x, init_y):
    self.top_body = world.CreateDynamicBody(
      position=(init_x, init_y - self.top_shift),
      angle=0.05 if self.left else -0.05,
      fixtures=self.top_fixture
    )

    self.bot_body = world.CreateDynamicBody(
      position=(init_x, init_y - self.bot_shift),
      angle=0.05 if self.left else -0.05,
      fixtures=self.bot_fixture
    )
    self.bot_body.ground_contact = False

    self.joint = world.CreateJoint(
      revoluteJointDef(
        bodyA=self.top_body,
        bodyB=self.bot_body,
        localAnchorA=(0, -self.top_height/2),
        localAnchorB=(0,  self.bot_height/2),
        enableMotor=True,
        enableLimit=True,
        maxMotorTorque=self.motors_torque,
        motorSpeed=1,
        lowerAngle=-1.6,
        upperAngle=-0.1
      )
    )

    self.top_body.color1 = self.color
    self.top_body.color2 = Color.BLACK
    self.bot_body.color1 = self.color
    self.bot_body.color2 = Color.BLACK

class RobotConfig:
  LEG_TOP_WIDTH  = 8.0
  LEG_TOP_HEIGHT = 34.0
  LEG_BOT_WIDTH  = 6.4
  LEG_BOT_HEIGHT = 34.0
  LIDAR_RANGE    = 160.0
  MOTORS_TORQUE  = 80.0
  SPEED_HIP      = 4.0
  SPEED_KNEE     = 6.0

  def __init__(self, scale, params=None):
    self.scale = scale

    self.hull_color = Color.rand()
    self.leg_color  = Color.rand()

    self.params = params if params is not None else np.ones(12)
    self.leg1_top_width  = self.params[0]  * self.LEG_TOP_WIDTH 
    self.leg1_top_height = self.params[1]  * self.LEG_TOP_HEIGHT
    self.leg1_bot_width  = self.params[2]  * self.LEG_BOT_WIDTH 
    self.leg1_bot_height = self.params[3]  * self.LEG_BOT_HEIGHT
    self.leg2_top_width  = self.params[4]  * self.LEG_TOP_WIDTH 
    self.leg2_top_height = self.params[5]  * self.LEG_TOP_HEIGHT
    self.leg2_bot_width  = self.params[6]  * self.LEG_BOT_WIDTH 
    self.leg2_bot_height = self.params[7]  * self.LEG_BOT_HEIGHT
    self.lidar_range     = self.params[8]  * self.LIDAR_RANGE
    self.motors_torque   = self.params[9]  * self.MOTORS_TORQUE
    self.speed_hip       = self.params[10] * self.SPEED_HIP
    self.speed_knee      = self.params[11] * self.SPEED_KNEE

  @classmethod
  def sample(cls, scale, np_random, low=0.5, high=1.5, symmetric=True):
    if symmetric:
      shape_params = np_random.uniform(low, high, size=4)
      shape_params = np.concatenate((shape_params, shape_params))
      dyna_params = np_random.uniform(low, high, size=4)
      params = np.concatenate((shape_params, dyna_params))
    else:
      params = np_random.uniform(low, high, size=12)
    return RobotConfig(scale, params=params)

class BipedalRobot:
  def __init__(self, config):
    self.config = config
    self.world = None

    self.hull = Hull(config)
    self.lidar = Lidar(config)

    self.joint1 = None
    self.joint2 = None

    self.leg1 = Leg(config, left=True)
    self.leg2 = Leg(config, left=False)

  @property
  def parts(self):
    # NOTE: their order is important for rendering.
    return self.leg1.parts + self.leg2.parts + self.hull.parts

  @property
  def joints(self):
    # NOTE: their order defines the action space.
    return [self.joint1, self.leg1.joint, self.joint2, self.leg2.joint]

  def destroy(self):
    if self.world is not None:
      for part in self.parts:
        if part is not None:
          self.world.DestroyBody(part)

  def reset(self, world, init_x, init_y, noise):
    self.world = world
    self.hull.reset(world, init_x, init_y, noise)
    self.leg1.reset(world, init_x, init_y)
    self.leg2.reset(world, init_x, init_y)
    self.lidar.reset()
    self.joint1 = world.CreateJoint(
      revoluteJointDef(
        bodyA=self.hull.body,
        bodyB=self.leg1.top_body,
        localAnchorA=(0, self.leg1.leg_down),
        localAnchorB=(0, self.leg1.top_height / 2),
        enableMotor=True,
        enableLimit=True,
        maxMotorTorque=self.config.motors_torque,
        motorSpeed=-1.0,
        lowerAngle=-0.8,
        upperAngle=1.1
      )
    )
    self.joint2 = world.CreateJoint(
        revoluteJointDef(
        bodyA=self.hull.body,
        bodyB=self.leg2.top_body,
        localAnchorA=(0, self.leg2.leg_down),
        localAnchorB=(0, self.leg2.top_height / 2),
        enableMotor=True,
        enableLimit=True,
        maxMotorTorque=self.config.motors_torque,
        motorSpeed=1.0,
        lowerAngle=-0.8,
        upperAngle=1.1
      )
    )

  def step(self, action):
    joint0, joint1, joint2, joint3 = self.joints
    joint0.motorSpeed     = self.config.speed_hip     * np.sign(action[0])
    joint0.maxMotorTorque = self.config.motors_torque * np.clip(np.abs(action[0]), 0, 1)
    joint1.motorSpeed     = self.config.speed_knee    * np.sign(action[1])
    joint1.maxMotorTorque = self.config.motors_torque * np.clip(np.abs(action[1]), 0, 1)
    joint2.motorSpeed     = self.config.speed_hip     * np.sign(action[2])
    joint2.maxMotorTorque = self.config.motors_torque * np.clip(np.abs(action[2]), 0, 1)
    joint3.motorSpeed     = self.config.speed_knee    * np.sign(action[3])
    joint3.maxMotorTorque = self.config.motors_torque * np.clip(np.abs(action[3]), 0, 1)
    return self.joints

  def scan(self, world, pos):
    self.lidar.scan(world, pos)
    return self.lidar.callbacks