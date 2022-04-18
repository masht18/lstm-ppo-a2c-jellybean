import numpy as np
import pandas as pd
from torch import log
import torch.nn
import torch.optim
import torch.nn.functional as F
from torch.distributions import Categorical

from gym.wrappers import LazyFrames
from gym.spaces import Box

from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    '''The agent class that is to be filled.
       You are allowed to add any method you
       want to this class.
    '''

    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.lz4_compress = False

        self.feature_pool = torch.nn.MaxPool2d(5)
        #self.actor_lstm = torch.nn.LSTM(15*15*4+3+27, 64, num_layers=2).to(device)
        self.actor = torch.nn.Sequential(torch.nn.Linear(3*3*4+3+27, 32),
                                          torch.nn.Tanh(),
                                         torch.nn.Linear(32, 32),
                                        torch.nn.Tanh(),
                                         torch.nn.Linear(32, 4), 
                                         torch.nn.Softmax(dim=1)).to(device)
        self.critic = torch.nn.Sequential(torch.nn.Linear(3*3*4+3+27, 32),
                                          torch.nn.Tanh(),
                                          torch.nn.Linear(32, 32),
                                          torch.nn.Tanh(), 
                                          torch.nn.Linear(32, 1)).to(device)

        self.optimizer = torch.optim.Adam([
                        {'params': self.actor.parameters(), 'lr': 0.0003},
                        {'params': self.critic.parameters(), 'lr': 0.001}
                    ])
        self.MseLoss = torch.nn.MSELoss()


        self.discount = 0.99

        self.num_stack = 2000
        self.window = int(self.num_stack/2)
        self.step_freq = 100
        self.t = 0
        self.eps_clip = 0.2

        self.vision_frames = deque(maxlen=self.num_stack)
        self.scent_frames = deque(maxlen=self.num_stack)
        self.feature_frames = deque(maxlen=self.num_stack)
        self.rewards = deque(maxlen=self.num_stack)
        self.actions = deque(maxlen=self.num_stack)
        
        self.old_log_probs = torch.zeros(self.window).to(device)

    def load_weights(self, root_path):
        #lstm_weights = os.path.join(root_path, 'policy_lstm.pth')
        #self.actor_lstm.load_state_dict(torch.load(policy_weights))
        
        actor_weights = os.path.join(root_path, 'actor.pth')
        self.actor.load_state_dict(torch.load(actor_weights))
        
        critic_weights = os.path.join(root_path, 'critic.pth')
        self.critic.load_state_dict(torch.load(policy_weights))
        

    def act(self, curr_obs, mode='eval'):
        if curr_obs==None:
            return self.env_specs['action_space'].sample()
        curr_scent, curr_vision, curr_features, _ = curr_obs
        curr_features = self.pool_features(curr_features)
        curr_vision = self.pool_vision(curr_vision)
        
        probs = self.policy_function(torch.tensor(curr_features), torch.tensor(curr_scent), torch.tensor(curr_vision))
        
        if mode == 'eval':
            return torch.argmax(probs)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            return action

    def update(self, curr_obs, action, reward, next_obs, done, timestep):
        if curr_obs == None:
            print("No observation")
            return
        
        curr_scent, curr_vision, curr_features,  _ = curr_obs
        next_scent, curr_vision, next_features, _ = next_obs
        
        next_features = self.pool_features(next_features)
        curr_features = self.pool_features(curr_features)
        
        curr_vision = self.pool_vision(curr_vision)
        
        self.vision_frames.append(curr_vision)
        self.scent_frames.append(curr_scent)
        self.feature_frames.append(curr_features)
        self.rewards.append(reward)
        self.actions.append(action)
        self.t += 1
        
        if self.t == self.num_stack:
            value = self.critic_function(self._tensor_from_queue(self.feature_frames)[:self.window], 
                                         self._tensor_from_queue(self.scent_frames)[:self.window],
                                        self._tensor_from_queue(self.vision_frames)[:self.window])
            probs = self.policy_function(self._tensor_from_queue(self.feature_frames)[:self.window], 
                                         self._tensor_from_queue(self.scent_frames)[:self.window],
                                         self._tensor_from_queue(self.vision_frames)[:self.window])
            dist = torch.distributions.Categorical(probs=probs)
            dist_entropy = dist.entropy()
            
            rewards = self.calculate_return()
            advantage = rewards - value.detach()
            
            ratios = torch.exp(dist.log_prob(self._tensor_from_queue(self.actions)[:self.window]) - self.old_log_probs)
            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantage

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(value, rewards) - 0.01*dist_entropy
            loss = loss.mean()
            #print(loss)
            
            # store the old log prob estimates for next actions
            with torch.no_grad():
                next_probs = self.policy_function(self._tensor_from_queue(self.feature_frames)[self.window:], 
                                             self._tensor_from_queue(self.scent_frames)[self.window:],
                                             self._tensor_from_queue(self.vision_frames)[self.window:])
                dist = torch.distributions.Categorical(probs=next_probs)

                self.old_log_probs = dist.log_prob(self._tensor_from_queue(self.actions)[self.window:])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.t = self.window
            
            #if timestep > 4990 and timestep < 5000:
            #    print(probs)
            
        if done:
            self.model_save()
            

    def calculate_grid():
        pass
    
    def model_save(self):
        #PATH = 'policy_lstm.pth'
        #torch.save(self.actor_lstm.state_dict(), PATH)
        
        PATH = 'actor.pth'
        torch.save(self.actor.state_dict(), PATH)
        
        PATH = 'critic.pth'
        torch.save(self.critic.state_dict(), PATH)
    
    def pool_features(self, features):
        features = features.reshape(15, 15, 4)
        features = torch.from_numpy(features).permute(2, 0, 1)
        features = self.feature_pool(features)
        return np.array(features.flatten())
    
    def pool_vision(self, vision):
        vision = torch.from_numpy(vision).permute(2, 0, 1)
        vision = self.feature_pool(vision)
        return np.array(vision.flatten())

    def policy_function(self, features, scents, vision):
        #scents = torch.from_numpy(scents).to(device).squeeze()
        #print(features)
        #print(scents)
        if scents.shape[0] == 3:
            inputs = torch.cat([features, scents, vision]).unsqueeze(0).to(device)
        else:
            inputs = torch.cat([features, scents, vision], dim=1).to(device)

        #h, _ = self.actor_lstm(inputs.float())
        #print(h)
        output = self.actor(inputs.float())

        return torch.squeeze(output)
    
    def critic_function(self, features, scents, vision):
        inputs = torch.cat([features, scents, vision], dim=1).unsqueeze(1).to(device)

        output = self.critic(inputs.float())
        
        return torch.squeeze(output)

    def calculate_return(self):
        discounts = torch.tensor([self.discount**i for i in range(self.window)]).to(device)
        return self._tensor_from_queue(self.rewards)[:self.window] * discounts
    
    def _tensor_from_queue(self, queue):
        return torch.tensor(queue).to(device)

