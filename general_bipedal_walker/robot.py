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

class Hull:
  _POLY = [
    (-30,  9), 
    (  6,  9), 
    ( 34,  1),
    ( 34, -8), 
    (-30, -8)
  ]
  _COLOR_1 = (0.5, 0.4, 0.9)
  _COLOR_2 = (0.3, 0.3, 0.5)

  def __init__(
      self, 
      world
  ):
    self.world = world
    self.body = None
    self.fixture = fixtureDef(
      shape=polygonShape(
        vertices=[
          (x / world.scale, y / world.scale) 
          for x, y in Hull._POLY
        ]
      ),
      density=5.0,
      friction=0.1,
      categoryBits=0x0020,
      maskBits=0x001, # collide only with ground
      restitution=0.0 # 0.99 bouncy
    )

  @property
  def parts(self):
    return [self.body]

  def reset(self, init_x, init_y, noise):
    self.body = self.world.world.CreateDynamicBody(
      position=(init_x, init_y), 
      fixtures=self.fixture
    )
    self.body.color1 = Hull._COLOR_1
    self.body.color2 = Hull._COLOR_2
    self.body.ApplyForceToCenter(noise, True)

class Lidar:
  class Callback(rayCastCallback):
    def ReportFixture(self, fixture, point, normal, fraction):
      if (fixture.filterData.categoryBits & 1) == 0:
        return 1
      self.p2 = point
      self.fraction = fraction
      return 0

  def __init__(self, scan_range):
    self.scan_range = scan_range
    self.callbacks = None

  def reset(self):
    self.callbacks = [Lidar.Callback() for _ in range(10)]

  def scan(self, pos, world):
    for i, lidar in enumerate(self.callbacks):
      self.callbacks[i].fraction = 1.0
      self.callbacks[i].p1 = pos
      self.callbacks[i].p2 = (
        pos[0] + math.sin(1.5*i/10.0) * self.scan_range,
        pos[1] - math.cos(1.5*i/10.0) * self.scan_range
      )
      world.world.RayCast(lidar, lidar.p1, lidar.p2)

class Leg:
  _COLOR_1 = np.array([0.6, 0.3, 0.5])
  _COLOR_2 = np.array([0.4, 0.2, 0.3])

  def __init__(
      self, 
      world,
      top_width,
      top_height,
      bot_width,
      bot_height,
      motors_torque,
      right=False
  ):
    self.world = world
    self.right = right
    self.motors_torque = motors_torque
    self.leg_down = -8 / world.scale

    self.top_width = top_width / world.scale
    self.top_height = top_height / world.scale
    self.top_shift = self.top_height / 2 + self.leg_down
    self.top_body = None
    self.top_fixture = fixtureDef(
      shape=polygonShape(
        box=(
          self.top_width / 2, 
          self.top_height / 2
        )
      ),
      density=1.0,
      restitution=0.0,
      categoryBits=0x0020,
      maskBits=0x001
    )

    self.joint = None

    self.bot_width = bot_width / world.scale
    self.bot_height = bot_height / world.scale
    self.bot_shift = self.top_height + self.bot_height / 2 + self.leg_down
    self.bot_body = None
    self.bot_fixture = fixtureDef(
      shape=polygonShape(
        box=(
          self.bot_width / 2, 
          self.bot_height / 2
        )
      ),
      density=1.0,
      restitution=0.0,
      categoryBits=0x0020,
      maskBits=0x001
    )

  @property
  def parts(self):
    return [self.top_body, self.bot_body]

  def reset(self, init_x, init_y):
    self.top_body = self.world.world.CreateDynamicBody(
      position=(init_x, init_y - self.top_shift),
      angle=-0.05 if self.right else 0.05,
      fixtures=self.top_fixture
    )

    self.bot_body = self.world.world.CreateDynamicBody(
      position=(init_x, init_y - self.bot_shift),
      angle=-0.05 if self.right else 0.05,
      fixtures=self.bot_fixture
    )
    self.bot_body.ground_contact = False

    self.joint = self.world.world.CreateJoint(
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

    shade = -0.1 if self.right else 0.1
    self.top_body.color1 = tuple(Leg._COLOR_1 + shade)
    self.top_body.color2 = tuple(Leg._COLOR_2 + shade)
    self.bot_body.color1 = tuple(Leg._COLOR_1 + shade)
    self.bot_body.color2 = tuple(Leg._COLOR_2 + shade)

class RobotConfig:
  LEG_TOP_WIDTH  = 8.0
  LEG_TOP_HEIGHT = 34.0
  LEG_BOT_WIDTH  = 6.4
  LEG_BOT_HEIGHT = 34.0
  LIDAR_RANGE    = 160.0
  MOTORS_TORQUE  = 80.0
  SPEED_HIP      = 4.0
  SPEED_KNEE     = 6.0

  def __init__(self, params=None):
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

  def sample(self):
    raise NotImplementedError

class BipedalRobot:
  def __init__(self, world, config):
    self.world = world
    self.config = config

    self.hull = Hull(world)
    self.lidar = Lidar(config.lidar_range)

    self.joint1 = None
    self.joint2 = None

    self.leg1 = Leg(
      world, 
      config.leg1_top_width,
      config.leg1_top_height,
      config.leg1_bot_width,
      config.leg1_bot_height,
      config.motors_torque,
      right=False
    )
    self.leg2 = Leg(
      world, 
      config.leg2_top_width,
      config.leg2_top_height,
      config.leg2_bot_width,
      config.leg2_bot_height,
      config.motors_torque,
      right=True
    )

  @property
  def parts(self):
    # NOTE: their order is important for rendering.
    return self.leg1.parts + self.leg2.parts + self.hull.parts

  @property
  def joints(self):
    # NOTE: their order defines the action space.
    return [self.joint1, self.leg1.joint, self.joint2, self.leg2.joint]

  def destroy(self):
    for part in self.parts:
      if part is not None:
        self.world.world.DestroyBody(part)

  def reset(self, init_x, init_y, noise):
    self.hull.reset(init_x, init_y, noise)
    self.leg1.reset(init_x, init_y)
    self.leg2.reset(init_x, init_y)
    self.lidar.reset()
    self.joint1 = self.world.world.CreateJoint(
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
    self.joint2 = self.world.world.CreateJoint(
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

  def scan(self, pos):
    self.lidar.scan(pos, self.world)
    return self.lidar.callbacks