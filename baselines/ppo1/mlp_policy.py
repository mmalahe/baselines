from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        ac_avail = U.get_placeholder(name="acavail", dtype=tf.float32, shape=[sequence_length] + list(ac_space.shape))
        
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
        
        # Gated probabilities implementation
        pdparam_prob = tf.exp(pdparam)/(tf.exp(pdparam)+1.0)       
        
        pd_gated_raw = tf.multiply(ac_avail, pdparam_prob)
        pd_gated_totals = tf.reduce_sum(pd_gated_raw,axis=1)
        pd_gated = pd_gated_raw/tf.reshape(pd_gated_totals, (-1, 1))
        
        pd_gated = tf.clip_by_value(pd_gated, 0.00001, 0.9999) # keep probability non-zero to fend of them nans
        
        pd_gated_logit = tf.log(pd_gated/(1.0-pd_gated))
        
        self.pd = pdtype.pdfromflat(pd_gated_logit)
        
        # Standard implementation
        #~ self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob, ac_avail], [ac, self.vpred])

    def act(self, stochastic, ob, ac_avail):
        ac1, vpred1 =  self._act(stochastic, ob[None], ac_avail[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

