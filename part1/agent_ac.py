import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

class Policy_ac(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()
        self.entropy_beta = 0.01 # small coefficient to keep exploration alive without dominating the loss

        """
            Actor network
        """       
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)
        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        """
            Critic
        """
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value = self.fc3_critic_value(x_critic)
     
        return normal_dist, value

class Agent_ac(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        # separate parameter groups so I can give the critic a higher lr
        # critic needs to converge faster to give the actor a reliable baseline
        actor_params = [p for name, p in self.policy.named_parameters() if 'actor' in name or 'sigma' in name]
        critic_params = [p for name, p in self.policy.named_parameters() if 'critic' in name]

        self.optimizer = torch.optim.Adam([
            {'params': actor_params, 'lr': 3e-4},   # actor learns slower — more stable policy updates
            {'params': critic_params, 'lr': 1e-3}   # critic learns faster — needs a good value estimate quickly
        ])

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.7)

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #        

        # run both s and s' through the critic to get V(s) and V(s')
        _, state_values = self.policy(states)
        _, next_state_values = self.policy(next_states)
        state_values = state_values.squeeze(-1)
        next_state_values = next_state_values.squeeze(-1)

        # TD target: r + gamma * V(s') — zero out V(s') on terminal steps
        td_target = rewards + self.gamma * next_state_values.detach() * (1 - done)
        advantage = td_target.detach() - state_values

        # Normalizing advantage (not rewards) keeps the actor update scale consistent
        # Across episodes without distorting the reward signal the critic sees
        # This stabilizes the Actor's learning steps.
        advantage_norm = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Calculate True Entropy (Positive value)
        true_entropy = torch.stack([dist.entropy().mean() for dist, _ in [self.policy(s.unsqueeze(0)) for s in states]]).mean()

        # critic loss: MSE between predicted V(s) and the TD target
        critic_loss = F.mse_loss(state_values, td_target.detach()) 
        
        # actor_loss: subtract the true entropy to maximize it (encouraging exploration)
        actor_loss = -(action_log_probs * advantage_norm.detach()).mean() - (self.policy.entropy_beta * true_entropy)
                
        total_loss = actor_loss + critic_loss
        self.scheduler.step() 

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)  # clip to avoid exploding gradients
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), true_entropy.item()

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist, _ = self.policy(x)

        if evaluation:  
            return normal_dist.mean, None
        else:     # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)