import gym
import sunblaze_envs


if __name__ == '__main__':
    for env in sorted([env.id for env in sunblaze_envs.registry.all()]):
        if not env.startswith('Sunblaze'):
            continue

        print(env)
