"""Based on a2c_episodes/base.py, but modified to handle varying obs length for EPOpt"""

import os
import gym

import numpy as np
import tensorflow as tf
from baselines.common.distributions import make_pdtype
from baselines.a2c.utils import fc, seq_to_batch, batch_to_seq, ortho_init, lstm

def make_env(env_id, process_idx=0, outdir=None):
    import sunblaze_envs

    env = sunblaze_envs.make(env_id)
    if outdir:
        env = sunblaze_envs.MonitorParameters(
            env,
            output_filename=os.path.join(outdir, 'env-parameters-{}.json'.format(process_idx))
        )

    return env


class lstm_policy(object):
    """ Creates policy and value LSTM networks, with parameter sharing.
    There is one hidden layer with nlstm units, default 256.
    Both discrete and continuous action spaces are supported.
    Environments with a discrete action space have a softmax policy, while environments with
    a continuous action space have Gaussian with diagonal covariance. """

    def __init__(self, sess, ob_space, ac_space, nenvs, nsteps, nlstm=256, reuse=False, feature_mlp=True):

        # Here the batch size is 1, i.e. one trajectory
        # also assume nenvs=1
        if nsteps is None:
            ob_shape = (None,)+ob_space.shape
            M = tf.placeholder(tf.float32, [None])
        else:
            ob_shape = (nsteps,)+ob_space.shape
            M = tf.placeholder(tf.float32, [nsteps])

        if len(ac_space.shape) == 0:
            # discrete set of actions
            nact = ac_space.n
            discrete = True
        else:
            actdim = ac_space.shape[0]
            discrete = False
        X = tf.placeholder(tf.float32, ob_shape, name="Ob")
        S = tf.placeholder(tf.float32, [1, nlstm*2]) # states
        
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            if feature_mlp:
                h1 = activ(fc(X, "fc1", nh=nlstm, init_scale=np.sqrt(2)))
                h2 = activ(fc(h1, "fc2", nh=nlstm, init_scale=np.sqrt(2)))
                xs = batch_to_seq(h2, 1, nsteps)
            else:
                xs = batch_to_seq(X, 1, nsteps)
            ms = batch_to_seq(M, 1, nsteps)
            h5, snew = lstm(xs, ms, S, "lstm1", nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, "vf", 1)
            if discrete:
                pi = fc(h5, "pi", nact, init_scale=0.01)
            else:
                pi = fc(h5, "pi", actdim, init_scale=0.01)
                logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())

        self.pdtype = make_pdtype(ac_space)
        if discrete:
            self.pd = self.pdtype.pdfromflat(pi)
        else:
            pdparam = tf.concat([pi, pi*0.0+logstd], axis=1)
            self.pd = self.pdtype.pdfromflat(pdparam)

        v0 = vf[:,0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((1, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
