import torch

class RolloutBuffer:

    def __init__(self, max_size, device, obs_dims, act_dims, stores_costs):
        self.max_size = max_size
        self.device = device
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.ptr = 0

        self.obs_buffer = torch.empty((max_size, *obs_dims), dtype=torch.float32, device=self.device)
        self.act_buffer = torch.empty((max_size, *act_dims), dtype=torch.float32, device=self.device)
        self.reward_buffer = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)
        self.stores_costs = stores_costs
        if stores_costs:
            self.cost_buffer = torch.empty((max_size, 1),dtype=torch.float32, device=self.device)


    def store(self, obs, act, reward, cost=None):
        t = self.ptr
        if t >= self.max_size:
            raise RuntimeError('RolloutBuffer overflow')

        self.obs_buffer[t].copy_(obs)
        self.act_buffer[t].copy_(act)
        self.reward_buffer[t] = reward
        if self.stores_costs and cost is not None:
            self.cost_buffer[t] = cost

        self.ptr = t + 1

    def query(self):
        pass