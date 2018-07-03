import numpy as np
import torch
import random
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from utils import evaluate_actions_sil

# replay buffer...
class ReplayBuffer:
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)
    
    def add(self, obs_t, action, R):
        data = (obs_t, action, R)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    
    def _encode_sample(self, idxes):
        obses_t, actions, returns = [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, R = data
            obses_t.append(obs_t)
            actions.append(action)
            returns.append(R)
        return np.array(obses_t), np.array(actions), np.array(returns)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 1e-6)
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size, beta):
        idxes = self._sample_proportional(batch_size)
        if beta > 0:
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self._storage)) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            weights = np.ones_like(idxes, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

# self-imitation learning
class sil_module:
    def __init__(self, network, args, optimizer):
        self.args = args
        self.network = network
        self.running_episodes = [[] for _ in range(self.args.num_processes)]
        self.optimizer = optimizer
        self.buffer = PrioritizedReplayBuffer(self.args.capacity, self.args.sil_alpha)
        # some other parameters...
        self.total_steps = []
        self.total_rewards = []

    # add the batch information into it...
    def step(self, obs, actions, rewards, dones):
        for n in range(self.args.num_processes):
            self.running_episodes[n].append([obs[n], actions[n], rewards[n]])
        # to see if can update the episode...
        for n, done in enumerate(dones):
            if done:
                self.update_buffer(self.running_episodes[n])
                self.running_episodes[n] = []
    
    # train the sil model...
    def train_sil_model(self):
        for n in range(self.args.n_update):
            obs, actions, returns, weights, idxes = self.sample_batch(self.args.batch_size)
            mean_adv, num_valid_samples = 0, 0
            if obs is not None:
                # need to get the masks
                # get basic information of network..
                obs = torch.tensor(obs, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
                returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
                weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
                max_nlogp = torch.tensor(np.ones((len(idxes), 1)) * self.args.max_nlogp, dtype=torch.float32)
                if self.args.cuda:
                    obs = obs.cuda()
                    actions = actions.cuda()
                    returns = returns.cuda()
                    weights = weights.cuda()
                    max_nlogp = max_nlogp.cuda()
                # start to next...
                value, pi = self.network(obs)
                action_log_probs, dist_entropy = evaluate_actions_sil(pi, actions)
                action_log_probs = -action_log_probs
                clipped_nlogp = torch.min(action_log_probs, max_nlogp)
                # process returns
                advantages = returns - value
                advantages = advantages.detach()
                masks = (advantages.cpu().numpy() > 0).astype(np.float32)
                # get the num of vaild samples
                num_valid_samples = np.sum(masks)
                num_samples = np.max([num_valid_samples, self.args.mini_batch_size])
                # process the mask
                masks = torch.tensor(masks, dtype=torch.float32)
                if self.args.cuda:
                    masks = masks.cuda()
                # clip the advantages...
                clipped_advantages = torch.clamp(advantages, 0, self.args.clip)
                mean_adv = torch.sum(clipped_advantages) / num_samples 
                mean_adv = mean_adv.item() 
                # start to get the action loss...
                action_loss = torch.sum(clipped_advantages * weights * clipped_nlogp) / num_samples
                entropy_reg = torch.sum(weights * dist_entropy * masks) / num_samples
                policy_loss = action_loss - entropy_reg * self.args.entropy_coef
                # start to process the value loss..
                # get the value loss
                delta = torch.clamp(value - returns, -self.args.clip, 0) * masks
                delta = delta.detach()
                value_loss = torch.sum(weights * value * delta) / num_samples
                total_loss = policy_loss + 0.5 * self.args.w_value * value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                # update the priorities
                self.buffer.update_priorities(idxes, clipped_advantages.squeeze(1).cpu().numpy())
        return mean_adv, num_valid_samples
    
    # update buffer
    def update_buffer(self, trajectory):
        positive_reward = False
        for (ob, a, r) in trajectory:
            if r > 0:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)
            self.total_steps.append(len(trajectory))
            self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            while np.sum(self.total_steps) > self.args.capacity and len(self.total_steps) > 1:
                self.total_steps.pop(0)
                self.total_rewards.pop(0)

    def add_episode(self, trajectory):
        obs = []
        actions = []
        rewards = []
        dones = []
        for (ob, action, reward) in trajectory:
            if ob is not None:
                obs.append(ob)
            else:
                obs.append(None)
            actions.append(action)
            rewards.append(np.sign(reward))
            dones.append(False)
        dones[len(dones) - 1] = True
        returns = self.discount_with_dones(rewards, dones, self.args.gamma)
        for (ob, action, R) in list(zip(obs, actions, returns)):
            self.buffer.add(ob, action, R)

    def fn_reward(self, reward):
        return np.sign(reward)

    def get_best_reward(self):
        if len(self.total_rewards) > 0:
            return np.max(self.total_rewards)
        return 0
    
    def num_episodes(self):
        return len(self.total_rewards)

    def num_steps(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if len(self.buffer) > 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size, beta=self.args.sil_beta)
        else:
            return None, None, None, None, None

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]
