from .registration import register, registry, make
from .monitor import MonitorParameters


# Classic control environments.

register(
    id='SunblazeCartPole-v0',
    entry_point='sunblaze_envs.classic_control:ModifiableCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='SunblazeCartPoleRandomNormal-v0',
    entry_point='sunblaze_envs.classic_control:RandomNormalCartPole',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='SunblazeCartPoleRandomExtreme-v0',
    entry_point='sunblaze_envs.classic_control:RandomExtremeCartPole',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='SunblazeMountainCar-v0',
    entry_point='sunblaze_envs.classic_control:ModifiableMountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='SunblazeMountainCarRandomNormal-v0',
    entry_point='sunblaze_envs.classic_control:RandomNormalMountainCar',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='SunblazeMountainCarRandomExtreme-v0',
    entry_point='sunblaze_envs.classic_control:RandomExtremeMountainCar',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='SunblazePendulum-v0',
    entry_point='sunblaze_envs.classic_control:ModifiablePendulumEnv',
    max_episode_steps=200,
)

register(
    id='SunblazePendulumRandomNormal-v0',
    entry_point='sunblaze_envs.classic_control:RandomNormalPendulum',
    max_episode_steps=200,
)

register(
    id='SunblazePendulumRandomExtreme-v0',
    entry_point='sunblaze_envs.classic_control:RandomExtremePendulum',
    max_episode_steps=200,
)

register(
    id='SunblazeAcrobot-v0',
    entry_point='sunblaze_envs.classic_control:ModifiableAcrobotEnv',
    max_episode_steps=500,
)

register(
    id='SunblazeAcrobotRandomNormal-v0',
    entry_point='sunblaze_envs.classic_control:RandomNormalAcrobot',
    max_episode_steps=500,
)

register(
    id='SunblazeAcrobotRandomExtreme-v0',
    entry_point='sunblaze_envs.classic_control:RandomExtremeAcrobot',
    max_episode_steps=500,
)

# Mujoco environments

register(
    id='SunblazeHopper-v0',
    entry_point='sunblaze_envs.mujoco:ModifiableRoboschoolHopper',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='SunblazeHopperRandomNormal-v0',
    entry_point='sunblaze_envs.mujoco:RandomNormalHopper',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='SunblazeHopperRandomExtreme-v0',
    entry_point='sunblaze_envs.mujoco:RandomExtremeHopper',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='SunblazeHalfCheetah-v0',
    entry_point='sunblaze_envs.mujoco:ModifiableRoboschoolHalfCheetah',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='SunblazeHalfCheetahRandomNormal-v0',
    entry_point='sunblaze_envs.mujoco:RandomNormalHalfCheetah',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='SunblazeHalfCheetahRandomExtreme-v0',
    entry_point='sunblaze_envs.mujoco:RandomExtremeHalfCheetah',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

#print(registration.registry)
#print(registration.registry.__dict__)

'''Commented-out code for registering unused environment combinations
CLASSIC_CONTROL = {
    'CartPole': [
        'StrongPush',
        'WeakPush',
        'ShortPole',
        'LongPole',
        'LightPole',
        'HeavyPole',
    ],
    'MountainCar': [
        'LowStart',
        'HighStart',
        'WeakForce',
        'StrongForce',
        'LightCar',
        'HeavyCar',
    ],
    'Pendulum': [
        'Light',
        'Heavy',
        'Short',
        'Long',
    ],
    'Acrobot': [
        'Light',
        'Heavy',
        'Short',
        'Long',
        'LowInertia',
        'HighInertia',
    ]
}

for baseline, variants in CLASSIC_CONTROL.items():
    for variant in variants:
        if baseline == 'CartPole':
            max_length = 200
            goal_achieved = 195.0
        elif baseline == 'MountainCar':
            max_length = 200
            goal_achieved = -110.0
        elif baseline == 'Pendulum':
            max_length = 200
            goal_achieved = None
        elif baseline == 'Acrobot':
            max_length = 500
            goal_achieved = None

        register(
            id='Sunblaze{}{}-v0'.format(baseline, variant),
            entry_point='sunblaze_envs.classic_control:{}{}'.format(variant, baseline),
            max_episode_steps=max_length,
            reward_threshold=goal_achieved,
        )

        register(
            id='Sunblaze{}Random{}-v0'.format(baseline, variant),
            entry_point='sunblaze_envs.classic_control:Random{}{}'.format(variant, baseline),
            max_episode_steps=max_length,
            reward_threshold=goal_achieved,
        )
'''

'''Commented-out code for registering unused environment combinations
MUJOCO = {
    'Hopper': [
        'Strong',
        'Weak',
        'HeavyTorso',
        'LightTorso',
        'SlipperyJoints',
        'RoughJoints',
    ],
    'HalfCheetah': [
        'Strong',
        'Weak',
        'HeavyTorso',
        'LightTorso',
        'SlipperyJoints',
        'RoughJoints',
    ]
}

for baseline, variants in MUJOCO.items():
    for variant in variants:
        if baseline == 'Hopper':
            goal_achieved = 3800.0
        elif baseline == 'HalfCheetah':
            goal_achieved = 4800.0
        # elif baseline == 'Ant':
            # goal_achieved = 6000.0

        register(
            id='Sunblaze{}{}-v0'.format(baseline, variant),
            entry_point='sunblaze_envs.mujoco:{}{}'.format(variant, baseline),
            max_episode_steps=1000,
            reward_threshold=goal_achieved,
        )

        register(
            id='Sunblaze{}Random{}-v0'.format(baseline, variant),
            entry_point='sunblaze_envs.mujoco:Random{}{}'.format(variant, baseline),
            max_episode_steps=1000,
            reward_threshold=goal_achieved,
        )
'''


'''Commented-out Ant (not used at the moment)

register(
    id='SunblazeAnt-v0',
    entry_point='sunblaze_envs.mujoco:ModifiableRoboschoolAnt',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='SunblazeAntRandomNormal-v0',
    entry_point='sunblaze_envs.mujoco:RandomNormalAnt',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='SunblazeAntRandomExtreme-v0',
    entry_point='sunblaze_envs.mujoco:RandomExtremeAnt',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
'''

'''Commented-out registration code for Atari and Doom envs

from .breakout import Breakout
from .space_invaders import SpaceInvaders
from .vizdoom import VizDoomEnvironment

# Maximum number of episode steps with frameskip of 4.
MAX_EPISODE_STEPS = 10000

def register_delayed_actions(env_id, entry_point, set_a, set_b, kwargs=None):
    """Helper for registering environment with delayed actions."""
    if kwargs is None:
        kwargs = {}

    for set_name, set_range in [('A', set_a), ('B', set_b)]:
        kwargs.update({
            'wrapped_class': entry_point,
            'wrappers': [
                ('sunblaze_envs.wrappers:ActionDelayWrapper', {
                    'delay_range_start': set_range[0],
                    'delay_range_end': set_range[1],
                }),
            ],
        })

        register(
            id='Sunblaze{}DelayedActionsSet{}-v0'.format(env_id, set_name),
            entry_point='sunblaze_envs.wrappers:wrap_environment',
            max_episode_steps=MAX_EPISODE_STEPS,
            kwargs=kwargs,
        )


# Physical 2D world environments.
for game in [Breakout, SpaceInvaders]:
    worlds = game.worlds.keys()
    game = game.__name__

    for world in worlds:
        if world == 'baseline':
            name = ''
        else:
            name = ''.join([w.capitalize() for w in world.split('_')])

        # Default frameskip (4) environment.
        register(
            id='Sunblaze{}{}-v0'.format(game, name),
            entry_point='sunblaze_envs:{}'.format(game),
            max_episode_steps=MAX_EPISODE_STEPS,
            kwargs={
                'world': world
            }
        )

        # No frameskip environment.
        register(
            id='Sunblaze{}{}NoFrameskip-v0'.format(game, name),
            entry_point='sunblaze_envs:{}'.format(game),
            max_episode_steps=4 * MAX_EPISODE_STEPS,
            kwargs={
                'world': world,
                'frameskip': 1,
            }
        )

        # Delayed action modification of the environment.
        if world == 'baseline':
            register_delayed_actions(
                env_id=game,
                entry_point='sunblaze_envs:{}'.format(game),
                set_a=(0, 3),
                set_b=(1, 5),
                kwargs={
                    'world': world
                }
            )

# VizDoom environments.
for scenario, variants in VizDoomEnvironment.scenarios.items():
    for name, variant in variants.items():
        scenario_name = ''.join([w.capitalize() for w in scenario.split('_')])

        if name == 'baseline':
            variant_name = ''
        else:
            variant_name = ''.join([w.capitalize() for w in name.split('_')])

        # Default frameskip (4) environment.
        register(
            id='SunblazeVizDoom{}{}-v0'.format(scenario_name, variant_name),
            entry_point='sunblaze_envs:VizDoomEnvironment',
            max_episode_steps=MAX_EPISODE_STEPS,
            kwargs={
                'scenario': scenario,
                'variant': name,
            }
        )

        # No frameskip environment.
        register(
            id='SunblazeVizDoom{}{}NoFrameskip-v0'.format(scenario_name, variant_name),
            entry_point='sunblaze_envs:VizDoomEnvironment',
            max_episode_steps=MAX_EPISODE_STEPS,
            kwargs={
                'scenario': scenario,
                'variant': name,
                'frameskip': 1,
            }
        )

        # Delayed action modification of the environment.
        if name == 'baseline':
            register_delayed_actions(
                env_id='VizDoom{}'.format(scenario_name),
                entry_point='sunblaze_envs:VizDoomEnvironment',
                set_a=(0, 5),
                set_b=(5, 10),
                kwargs={
                    'scenario': scenario,
                    'variant': name,
                }
            )
'''
