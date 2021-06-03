

class Trandition():
    def __init__(self, args):
        self.n_actions = args.n_actions
    
    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        pass

    def get_q_values(self, batch, max_episode_len):
        pass 

    def init_hidden(self, episode_num):
        pass

    def save_model(self, num):
        pass

    def get_model_idx(self):
        pass

    def load_model(self, rnn_root, vdn_root):
        pass

class Random(Trandition):
    def __init__(self, args):
        super(Random, self).__init__(args)
        print('Init alg random')

