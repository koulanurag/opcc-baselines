class Base:
    def __init__(self, obs_size, action_size, deterministic):
        self.__obs_size = obs_size
        self.__action_size = action_size
        self.__deterministic = deterministic

    @property
    def obs_size(self):
        return self.__obs_size

    @property
    def action_size(self):
        return self.__action_size

    @property
    def deterministic(self):
        return self.__deterministic
