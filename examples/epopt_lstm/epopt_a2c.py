"""(Further) adapted from a2c_baselines/a2c_episodes.py

Ideally, we should only have to modify the trajectory segment generation
"""

import os
import os.path as osp
import numpy as np
import time
import tensorflow as tf
import joblib
import logging
import pickle

#from baselines.a2c.utils import find_trainable_variables, Scheduler, make_path, discount_with_dones
#from baselines.a2c.utils import mse

from baselines import logger
from baselines.common import explained_variance
from baselines.a2c.utils import find_trainable_variables, Scheduler, make_path, discount_with_dones
from baselines.a2c.utils import mse

from ..ppo2_baselines.ppo2_episodes import constfn  # only needed for adaptive epsilon
from ..a2c_baselines.a2c_episodes import Runner as BaseRunner


class EPOptModel(object):
    """Modification of the Model class in a2c_episdoes.py which supports variable batch sizes for EPOpt"""

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            lr=7e-4, alpha=0.99, epsilon=1e-5, total_timesteps=int(20e6), lrschedule='linear'):

        '''
        sess = tf.get_default_session()
        nbatch = nenvs*nsteps

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])
        '''

        # begin diff
        sess = tf.get_default_session()

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, reuse=True)

        L = tf.placeholder(tf.int32, [1])
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        # end diff

        neglogpac = train_model.pd.neglogp(A) # length max_episode_steps
        pg_loss = tf.reduce_mean(tf.slice(ADV * neglogpac, [0], L))
        vf_loss = tf.reduce_mean(tf.slice(mse(tf.squeeze(train_model.vf), R), [0], L))
        entropy = tf.reduce_mean(tf.slice(train_model.pd.entropy(), [0], L))
        loss = pg_loss-entropy*ent_coef+vf_loss*vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values, length):
            advs = rewards-values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr, L:np.asarray([length])}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run([pg_loss, vf_loss, entropy, _train], td_map)
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class EPOptRunner(BaseRunner):
    """Modified the trajectory generator in A2C to follow EPOpt-e"""

    def run(self, *,
            # EPOpt specific - could go in __init__ but epsilon is callable
            paths, epsilon
            ):
        """Instead of doing a trajectory of nsteps (ie, "horizon"), do a
        sample N "paths" and then return the bottom epsilon-percentile
        """
        # FIXME(cpacker): currently only works with single-threading
        assert(self.env.num_envs==1)

        # Store all N trajectories sampled then return data of bottom-epsilon
        # lists -> lists of lists
        n_mb_obs, n_mb_rewards, n_mb_actions, n_mb_values, n_mb_dones = [[] for _ in range(paths)], [[] for _ in range(paths)], [[] for _ in range(paths)], [[] for _ in range(paths)], [[] for _ in range(paths)]
        num_episodes = 0
        mb_states = self.model.initial_state
        for N in range(paths):

            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = n_mb_obs[N], n_mb_rewards[N], n_mb_actions[N], n_mb_values[N], n_mb_dones[N]
            self.states = self.model.initial_state
            for _ in range(self.nsteps):
                actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_values.append(values)
                mb_dones.append(self.dones)
                obs, rewards, dones, _ = self.env.step(actions)
                self.states = states
                self.dones = dones
                #for i, done in enumerate(dones):
                    #if done:
                        #self.obs[i] = self.obs[i]*0
                self.obs = obs
                mb_rewards.append(rewards)
                # We only want to do one episode
                if self.dones:
                    break
            mb_dones.append(self.dones)

        # Compute the worst epsilon paths and concatenate them
        # TODO(cpacker): check - do this before or after discounting?
        episode_returns = [sum(r) for r in n_mb_rewards]
        cutoff = np.percentile(episode_returns, 100*epsilon)
        # indexes = [i for i, r in enumerate(mb_rewards) if r <= cutoff]

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_lens, mb_masks = [],[],[],[],[],[],[]
        for N in range(paths):
            #if n_mb_rewards[N] <= cutoff:
            if episode_returns[N] <= cutoff:
                # only count the episodes that are returned
                num_episodes += 1
                num_to_pad = self.nsteps-len(n_mb_rewards[N])
                mb_lens.append(len(n_mb_rewards[N]))
                # concatenate
                next_obs = np.asarray(n_mb_obs[N], dtype=np.float32).swapaxes(1,0).squeeze()
                next_rewards = np.asarray(n_mb_rewards[N], dtype=np.float32).swapaxes(1,0)
                if self.discrete:
                    next_actions = np.asarray(n_mb_actions[N], dtype=np.int32).swapaxes(1,0)
                else:
                    next_actions = np.asarray(n_mb_actions[N], dtype=np.float32).swapaxes(1,0)
                next_values = np.asarray(n_mb_values[N], dtype=np.float32).swapaxes(1,0)
                next_dones = np.asarray(n_mb_dones[N], dtype=np.bool).swapaxes(1,0)

                last_values = self.model.value(self.obs, self.states, self.dones).tolist() # should not matter what this value is
                # discount/bootstrap off value fn
                for n, (rewards, dones, value) in enumerate(zip(next_rewards, next_dones[:,1:], last_values)):
                    rewards = rewards.tolist()
                    dones = dones.tolist()
                    if dones[-1] == 0:
                        rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                    else:
                        rewards = discount_with_dones(rewards, dones, self.gamma)
                    next_rewards = np.asarray(rewards).flatten()

                mb_obs.append(np.concatenate((next_obs, np.zeros((num_to_pad,) + self.env.observation_space.shape, dtype=self.obs.dtype))))
                mb_rewards.append(np.concatenate((next_rewards, np.zeros(num_to_pad))))
                if self.discrete:
                    mb_actions.append(np.concatenate((next_actions.reshape(next_rewards.shape), np.zeros(num_to_pad, dtype=np.int32))))
                else:
                    mb_actions.append(np.concatenate((next_actions.reshape(next_rewards.shape[0], self.ac_space.shape[0]), np.zeros((num_to_pad,)+self.ac_space.shape, dtype=np.float32))))
                mb_values.append(np.concatenate((next_values.flatten(), np.zeros(num_to_pad))))
                mb_masks.append(np.concatenate((next_dones[:,:-1].flatten(), np.full(num_to_pad, True))))

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, num_episodes, mb_lens


def learn(policy, env,
          nsteps=5, total_episodes=int(10e3), max_timesteps=int(20e5),
          vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
          lr=7e-4, lrschedule='linear', epsilon=1e-5,
          alpha=0.99, gamma=0.99,
          save_interval=100, log_interval=100, keep_all_ckpt=False,
          paths=100, epopt_epsilon=1.0  # EPOpt specific
          ):

    # In the original paper, epsilon is fixed to 1.0 for the first 100
    # "iterations" before updating to desired value
    if isinstance(epopt_epsilon, float):
        epopt_epsilon = constfn(epopt_epsilon)
    else:
        assert callable(epopt_epsilon)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    max_episode_len = env.venv.envs[0].spec.max_episode_steps
    make_model = lambda: EPOptModel(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=max_episode_len, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=max_timesteps, lrschedule=lrschedule)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = EPOptRunner(env, model, nsteps=max_episode_len, gamma=gamma)

    nbatch = nenvs*nsteps
    tfirststart = time.time()
    update = 0
    episodes_so_far = 0
    old_savepath = None
    while True:
        update += 1
        if episodes_so_far >= total_episodes:
            break

        epsilonnow = epopt_epsilon(update)
        obs, states, rewards, masks, actions, values, num_episodes, lens = runner.run(paths=paths, epsilon=epsilonnow)
        episodes_so_far += num_episodes

        policy_loss = np.zeros(num_episodes)
        value_loss = np.zeros(num_episodes)
        policy_entropy = np.zeros(num_episodes)
        for i in range(num_episodes):
            policy_loss[i], value_loss[i], policy_entropy[i] = model.train(obs[i], states, rewards[i], masks[i], actions[i], values[i], lens[i])
        nseconds = time.time()-tfirststart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = np.mean([explained_variance(values[i], rewards[i]) for i in range(num_episodes)])
            logger.record_tabular("nupdates", update)
            logger.record_tabular("epsilon", epsilonnow)
            #logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("total_episodes", episodes_so_far)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", np.mean(policy_entropy))
            logger.record_tabular("value_loss", np.mean(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("time_elapsed", nseconds)
            logger.dump_tabular()

        if save_interval and logger.get_dir() and (update % save_interval == 0 or update == 1):
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            obs_norms = {}
            obs_norms['clipob'] = env.clipob
            obs_norms['mean'] = env.ob_rms.mean
            obs_norms['var'] = env.ob_rms.var+env.epsilon
            with open(osp.join(checkdir, 'normalize'), 'wb') as f:
                pickle.dump(obs_norms, f, pickle.HIGHEST_PROTOCOL) 
            model.save(savepath)

            if not keep_all_ckpt and old_savepath:
                print('Removing previous checkpoint', old_savepath)
                os.remove(old_savepath)
            old_savepath = savepath

    env.close()

