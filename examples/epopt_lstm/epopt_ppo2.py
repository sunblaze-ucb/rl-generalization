"""(Further) adapted from ppo2_baselines/ppo2_episodes.py

Ideally, we should only have to modify the trajectory segment generation
"""

import os
import time
import joblib
import numpy as np
import os.path as osp
import pickle
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance

from ..ppo2_baselines.ppo2_episodes import constfn, sf01
from ..ppo2_baselines.ppo2_episodes import Runner as BaseRunner


class EPOptModel(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        # begin diff
        train_model = policy(sess, ob_space, ac_space, nbatch_act, nsteps, reuse=True)
        # end diff

        L = tf.placeholder(tf.int32, [1])
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A) # length max_episode_steps
        # entropy is mean of action distribution over all time steps
        entropy = tf.reduce_mean(tf.slice(train_model.pd.entropy(), [0], L))

        vpred = train_model.vf # length max_episode_steps
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE) # length max_episode_steps
        vf_losses1 = tf.square(vpred - R) # length max_episode_steps
        vf_losses2 = tf.square(vpredclipped - R) # length max_episode_steps
        vf_loss = .5 * tf.reduce_mean(tf.slice(tf.reshape(tf.maximum(vf_losses1, vf_losses2), [-1]), [0], L))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac) # length max_episode_steps
        pg_losses = -ADV * ratio # length max_episode_steps
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE) # length max_episode_steps
        pg_loss = tf.reduce_mean(tf.slice(tf.maximum(pg_losses, pg_losses2), [0], L))
        approxkl = .5 * tf.reduce_mean(tf.slice(tf.square(neglogpac - OLDNEGLOGPAC), [0], L))
        clipfrac = tf.reduce_mean(tf.slice(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)), [0], L))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, length, states=None):
            advs = returns - values
            advs = (advs - advs[0:length].mean()) / (advs[0:length].std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values, L:np.asarray([length])}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101


class EPOptRunner(BaseRunner):
    """Modified the trajectory generator in PPO2 to follow EPOpt-e"""

    def run(self, *,
            # EPOpt specific - could go in __init__ but epsilon is callable
            paths, epsilon
            ):
        """Instead of doing a trajectory of nsteps (ie, "horizon"), do a
        sample N "paths" and then return the bottom epsilon-percentile
        """
        assert(self.env.num_envs==1)

        # Store all N trajectories sampled then return data of bottom-epsilon
        # lists -> lists of lists
        n_mb_obs, n_mb_rewards, n_mb_actions, n_mb_values, n_mb_dones, n_mb_neglogpacs = [[] for _ in range(paths)], [[] for _ in range(paths)], [[] for _ in range(paths)], [[] for _ in range(paths)], [[] for _ in range(paths)], [[] for _ in range(paths)]
        n_epinfos = [[] for _ in range(paths)]
        mb_states = self.model.initial_state
        num_episodes = 0

        for N in range(paths):
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, epinfos = n_mb_obs[N], n_mb_rewards[N], n_mb_actions[N], n_mb_values[N], n_mb_dones[N], n_mb_neglogpacs[N], n_epinfos[N]
            self.states = self.model.initial_state
            for _ in range(self.nsteps):  # nsteps == horizon == env.max_episode_steps
                actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
                mb_obs.append(self.obs.copy())
                mb_actions.append(actions)
                mb_values.append(values)
                mb_neglogpacs.append(neglogpacs)
                mb_dones.append(self.dones)
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo: epinfos.append(maybeepinfo)
                mb_rewards.append(rewards)
                # Stop once single thread has finished an episode
                if self.dones:  # ie [True]
                    break

        # Compute the worst epsilon paths and concatenate them
        episode_returns = [sum(r) for r in n_mb_rewards]
        cutoff = np.percentile(episode_returns, 100*epsilon)

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_lens = [],[],[],[],[],[],[]
        mb_returns, mb_advs = [],[]
        epinfos = []
        for N in range(paths):
            #if n_mb_rewards[N] <= cutoff:
            if episode_returns[N] <= cutoff:
                # only count and concatenate the episodes that are returned
                num_episodes += 1
                num_to_pad = self.nsteps-len(n_mb_rewards[N])
                # concatenate and pad with zeros
                mb_obs.append(np.concatenate((sf01(np.asarray(n_mb_obs[N], dtype=self.obs.dtype)), np.zeros((num_to_pad,) + self.env.observation_space.shape, dtype=self.model.train_model.X.dtype.name))))
                mb_rewards.append(np.concatenate((sf01(np.asarray(n_mb_rewards[N], dtype=np.float32)), np.zeros(num_to_pad))))
                if len(self.env.action_space.shape) == 0:
                    mb_actions.append(np.concatenate((sf01(np.asarray(n_mb_actions[N])), np.zeros(num_to_pad, dtype=np.int32))))
                else:
                    mb_actions.append(np.concatenate((sf01(np.asarray(n_mb_actions[N])), np.zeros((num_to_pad,)+self.env.action_space.shape, dtype=np.float32))))
                mb_values.append(np.concatenate((sf01(np.asarray(n_mb_values[N], dtype=np.float32)), np.zeros(num_to_pad))))
                mb_dones.append(np.concatenate((sf01(np.asarray(n_mb_dones[N], dtype=np.bool)), np.full(num_to_pad,True))))
                mb_neglogpacs.append(np.concatenate((sf01(np.asarray(n_mb_neglogpacs[N], dtype=np.float32)), np.zeros(num_to_pad))))
                epinfos.append(n_epinfos[N])
                mb_lens.append(len(n_mb_rewards[N]))

                last_values = self.model.value(self.obs, self.states, self.dones) # does not matter what this value is
                returns = np.zeros_like(n_mb_rewards[N])
                advs = np.zeros_like(n_mb_rewards[N])
                lastgaelam = 0
                for t in reversed(range(len(n_mb_rewards[N]))):
                    if t == len(n_mb_rewards[N]) - 1:
                        nextnonterminal = 1.0-self.dones # self.dones should always be [1]
                        nextvalues = last_values
                    else:
                        nextnonterminal = 1.0-n_mb_dones[N][t+1]
                        nextvalues = n_mb_values[N][t+1]
                    delta = n_mb_rewards[N][t] + self.gamma * nextvalues * nextnonterminal - n_mb_values[N][t]
                    advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
                returns = advs + np.asarray(n_mb_values[N], dtype=np.float32)
                mb_returns.append(np.concatenate((sf01(returns), np.zeros(num_to_pad))))
                mb_advs.append(advs)

        return (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs,
            mb_states, epinfos, num_episodes, mb_lens)


def learn(*, policy, env, nsteps, total_episodes, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, keep_all_ckpt=False,
            paths=100, epsilon=1.0  # EPOpt specific
            ):
    """Only difference here is that epsilon and N are specified and passed to
    runner.run()
    """

    # In the original paper, epsilon is fixed to 1.0 for the first 100
    # "iterations" before updating to desired value
    if isinstance(epsilon, float):
        epsilon = constfn(epsilon)
    else:
        assert callable(epsilon)

    # FIXME:
    # Callable lr and cliprange don't work (at the moment) with the
    # total_episodes terminating condition
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        raise NotImplementedError
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        raise NotImplementedError
        assert callable(cliprange)

    # total_timesteps = int(total_timesteps)
    total_episodes = int(total_episodes)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    max_episode_len = env.venv.envs[0].spec.max_episode_steps

    make_model = lambda : EPOptModel(policy=policy, ob_space=ob_space,
                    ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=max_episode_len, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = EPOptRunner(env=env, model=model, nsteps=max_episode_len, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    update = 0
    episodes_so_far = 0
    old_savepath = None
    while True:
        update += 1
        if episodes_so_far > total_episodes:
            break

        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        # frac = 1.0 - (update - 1.0) / nupdates
        frac = 1.0 - (update - 1.0) / total_episodes
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        epsilonnow = epsilon(update)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos, num_episodes, lens = runner.run(paths=paths, epsilon=epsilonnow) #pylint: disable=E0632
        # now all the other variables are lists of numpy arrays of length num_episodes

        episodes_so_far += num_episodes

        mblossvals = []
        if states is None: # nonrecurrent version
            raise NotImpelementedError("Use examples.epopt")
        else: # recurrent version
            for _ in range(noptepochs):
                '''
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))
                '''
                for i in range(num_episodes):
                    epinfobuf.extend(epinfos[i])
                    mblossvals.append(model.train(lrnow, cliprangenow, obs[i], returns[i], masks[i], actions[i], values[i], neglogpacs[i], lens[i], states))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = np.mean([explained_variance(values[i], returns[i]) for i in range(num_episodes)])
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("epsilon", epsilonnow)
            logger.logkv("total_episodes", episodes_so_far)
            #logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
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

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
