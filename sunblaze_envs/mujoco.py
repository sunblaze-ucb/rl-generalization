import contextlib
import os
import tempfile

import numpy as np
import xml.etree.ElementTree as ET

import roboschool
from roboschool.gym_mujoco_walkers import (
    RoboschoolForwardWalkerMujocoXML, RoboschoolHalfCheetah, RoboschoolHopper
)

from .base import EnvBinarySuccessMixin
from .classic_control import uniform_exclude_inner

# Determine Roboschool asset location based on its module path.
ROBOSCHOOL_ASSETS = os.path.join(roboschool.__path__[0], 'mujoco_assets')


class RoboschoolTrackDistSuccessMixin(EnvBinarySuccessMixin):
    """Treat reaching certain distance on track as a success."""

    def is_success(self):
         """Returns True is current state indicates success, False otherwise

         x=100 correlates to the end of the track on Roboschool,
         but with the default 1000 max episode length most (all?) agents
         won't reach it (DD PPO2 Hopper reaches ~40), so we use something lower
         """
         target_dist = 20
         if self.robot_body.pose().xyz()[0] >= target_dist:
             #print("[SUCCESS]: xyz is {}, reached x-target {}".format(
             #      self.robot_body.pose().xyz(), target_dist))
             return True
         else:
             #print("[NO SUCCESS]: xyz is {}, x-target is {}".format(
             #      self.robot_body.pose().xyz(), target_dist))
             return False 


class RoboschoolXMLModifierMixin:
    """Mixin with XML modification methods."""
    @contextlib.contextmanager
    def modify_xml(self, asset):
        """Context manager allowing XML asset modifcation."""

        # tree = ET.ElementTree(ET.Element(os.path.join(ROBOSCHOOL_ASSETS, asset)))
        tree = ET.parse(os.path.join(ROBOSCHOOL_ASSETS, asset))
        yield tree

        # Create a new temporary .xml file
        # mkstemp returns (int(file_descriptor), str(full_path))
        fd, path = tempfile.mkstemp(suffix='.xml')
        # Close the file to prevent a file descriptor leak
        # See: https://www.logilab.org/blogentry/17873
        # We can also wrap tree.write in 'with os.fdopen(fd, 'w')' instead
        os.close(fd)
        tree.write(path)

        # Delete previous file before overwriting self.model_xml
        if os.path.isfile(self.model_xml):
            os.remove(self.model_xml)
        self.model_xml = path

        # Original fix using mktemp:
        # mktemp (depreciated) returns str(full_path)
        #   modified_asset = tempfile.mktemp(suffix='.xml')
        #   tree.write(modified_asset)
        #   self.model_xml = modified_asset

    def __del__(self):
        """Deletes last remaining xml files after use"""
        # (Note: this won't ensure the final tmp file is deleted on a crash/SIGBREAK/etc.)
        if os.path.isfile(self.model_xml):
            os.remove(self.model_xml)


# Half Cheetah

class ModifiableRoboschoolHalfCheetah(RoboschoolHalfCheetah, RoboschoolTrackDistSuccessMixin):

    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500
   
    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 1.4

    RANDOM_LOWER_POWER = 0.7
    RANDOM_UPPER_POWER = 1.1
    EXTREME_LOWER_POWER = 0.5
    EXTREME_UPPER_POWER = 1.3

    def _reset(self, new=True):
        return super(ModifiableRoboschoolHalfCheetah, self)._reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }


class StrongHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, 'half_cheetah.xml', 'torso', action_dim=6, obs_dim=26, power=1.3)

    @property
    def parameters(self):
        parameters = super(StrongHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class WeakHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, 'half_cheetah.xml', 'torso', action_dim=6, obs_dim=26, power=0.5)

    @property
    def parameters(self):
        parameters = super(WeakHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class RandomStrongHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def randomize_power(self):
        self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomStrongHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomStrongHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters

class RandomWeakHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def randomize_power(self):
        self.power = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
            self.RANDOM_LOWER_POWER, self.EXTREME_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomWeakHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomWeakHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class HeavyTorsoHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.density = 1500
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(HeavyTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class LightTorsoHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.density = 500
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(LightTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomHeavyTorsoHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def randomize_mass(self):
        self.density = self.np_random.uniform(self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomHeavyTorsoHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomHeavyTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomLightTorsoHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def randomize_mass(self):
        self.density = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomLightTorsoHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomLightTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class SlipperyJointsHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.friction = 0.2
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(SlipperyJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RoughJointsHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.friction = 1.4
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(RoughJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomRoughJointsHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def randomize_friction(self):
        self.friction = self.np_random.uniform(self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomRoughJointsHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomRoughJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomSlipperyJointsHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):
    def randomize_friction(self):
        self.friction = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomSlipperyJointsHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomSlipperyJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomNormalHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):

    def randomize_env(self):
        self.density = self.np_random.uniform(self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = self.np_random.uniform(self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalHalfCheetah, self).parameters
        parameters.update({'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters


class RandomExtremeHalfCheetah(RoboschoolXMLModifierMixin, ModifiableRoboschoolHalfCheetah):

    def randomize_env(self):
        '''
        # self.armature = self.np_random.uniform(0.2, 0.5)
        self.density = self.np_random.uniform(self.LOWER_DENSITY, self.UPPER_DENSITY)
        self.friction = self.np_random.uniform(self.LOWER_FRICTION, self.UPPER_FRICTION)
        self.power = self.np_random.uniform(self.LOWER_POWER, self.UPPER_POWER)
        '''

        self.density = uniform_exclude_inner(self.np_random.uniform,
            self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = uniform_exclude_inner(self.np_random.uniform,
            self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.power = uniform_exclude_inner(self.np_random.uniform,
            self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomExtremeHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremeHalfCheetah, self).parameters
        parameters.update({'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters

# Hopper

class ModifiableRoboschoolHopper(RoboschoolHopper, RoboschoolTrackDistSuccessMixin):

    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 1.4

    RANDOM_LOWER_POWER = 0.6
    RANDOM_UPPER_POWER = 0.9
    EXTREME_LOWER_POWER = 0.4
    EXTREME_UPPER_POWER = 1.1

    def _reset(self, new=True):
        return super(ModifiableRoboschoolHopper, self)._reset()

    @property
    def parameters(self):
        return {'id': self.spec.id, }


class StrongHopper(ModifiableRoboschoolHopper):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, 'hopper.xml', 'torso', action_dim=3, obs_dim=15, power=1.1)

    @property
    def parameters(self):
        parameters = super(StrongHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class WeakHopper(ModifiableRoboschoolHopper):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, 'hopper.xml', 'torso', action_dim=3, obs_dim=15, power=0.4)

    @property
    def parameters(self):
        parameters = super(WeakHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class RandomStrongHopper(ModifiableRoboschoolHopper):
    def randomize_power(self):
        self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomStrongHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomStrongHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class RandomWeakHopper(ModifiableRoboschoolHopper):
    def randomize_power(self):
        self.power = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomWeakHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomWeakHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class HeavyTorsoHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def __init__(self):
        self.density = 1500
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(HeavyTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class LightTorsoHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def __init__(self):
        self.density = 500
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(LightTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomHeavyTorsoHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def randomize_mass(self):
        self.density = self.np_random.uniform(self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomHeavyTorsoHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomHeavyTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomLightTorsoHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def randomize_mass(self):
        self.density = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomLightTorsoHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomLightTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class SlipperyJointsHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def __init__(self):
        self.friction = 0.2
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(SlipperyJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RoughJointsHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def __init__(self):
        self.friction = 1.4
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(RoughJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomRoughJointsHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def randomize_friction(self):
        self.friction = self.np_random.uniform(self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomRoughJointsHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomRoughJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomSlipperyJointsHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):
    def randomize_friction(self):
        self.friction = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomSlipperyJointsHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomSlipperyJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomNormalHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):

    def randomize_env(self):
        self.density = self.np_random.uniform(self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = self.np_random.uniform(self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalHopper, self).parameters
        parameters.update({'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters


class RandomExtremeHopper(RoboschoolXMLModifierMixin, ModifiableRoboschoolHopper):

    def randomize_env(self):
        '''
        self.density = self.np_random.uniform(self.LOWER_DENSITY, self.UPPER_DENSITY)
        self.friction = self.np_random.uniform(self.LOWER_FRICTION, self.UPPER_FRICTION)
        self.power = self.np_random.uniform(self.LOWER_POWER, self.UPPER_POWER)
        '''

        self.density = uniform_exclude_inner(self.np_random.uniform,
            self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = uniform_exclude_inner(self.np_random.uniform,
            self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.power = uniform_exclude_inner(self.np_random.uniform,
            self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomExtremeHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremeHopper, self).parameters
        parameters.update({'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters
