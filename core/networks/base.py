class Base:
    def __init__(self, env_name, dataset_name, obs_size, action_size,
                 deterministic, constant_prior, prior_scale):
        self.__env_name = env_name
        self.__dataset_name = dataset_name
        self.__obs_size = obs_size
        self.__action_size = action_size
        self.__deterministic = deterministic
        self.__constant_prior = constant_prior
        self.__prior_scale = prior_scale

    @property
    def env_name(self):
        return self.__env_name

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def obs_size(self):
        return self.__obs_size

    @property
    def action_size(self):
        return self.__action_size

    @property
    def deterministic(self):
        return self.__deterministic

    @property
    def constant_prior(self):
        return self.__constant_prior

    @property
    def prior_scale(self):
        return self.__prior_scale
