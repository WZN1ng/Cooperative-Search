class Agent(object):
    def __init__(self, pos, view_range):
        self.pos = pos
        self.view_range = view_range
        self.time_step = 0
        self.reward = 0