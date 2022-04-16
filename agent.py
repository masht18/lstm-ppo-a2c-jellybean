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
        self.actor_lstm = torch.nn.LSTM(3*3*4+3+27, 16, num_layers=2).to(device)
        self.actor = torch.nn.Sequential(torch.nn.Linear(16, 4), 
                                         torch.nn.Softmax(dim=1)).to(device)
        self.critic = torch.nn.Sequential(torch.nn.Linear(3*3*4+3+27, 16),
                                          torch.nn.ReLU(), 
                                          torch.nn.Linear(16, 1)).to(device)

        self.optimizer_actor = torch.optim.Adam(list(self.actor.parameters()) + list(self.actor_lstm.parameters()), lr=0.001)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=0.01)

        #self.g_last_goal = np.zeros(11)

        self.discount = 0.7

        self.num_stack = 10
        self.step_freq = 100
        self.t = 0

        self.vision_frames = deque(maxlen=self.num_stack)
        self.scent_frames = deque(maxlen=self.num_stack)
        self.feature_frames = deque(maxlen=self.num_stack)
        self.rewards = deque(maxlen=self.num_stack)
        self.actions = deque(maxlen=self.num_stack)
        
        self.vision_frames_eval = deque(maxlen=self.num_stack)
        self.scent_frames_eval = deque(maxlen=self.num_stack)
        self.feature_frames_eval = deque(maxlen=self.num_stack)
        
        self.actor_loss = 0
        self.critic_loss = 0

    def load_weights(self, root_path):
        lstm_weights = os.path.join(root_path, 'policy_lstm.pth')
        self.actor_lstm.load_state_dict(torch.load(policy_weights))
        
        actor_weights = os.path.join(root_path, 'actor.pth')
        self.actor.load_state_dict(torch.load(actor_weights))
        
        critic_weights = os.path.join(root_path, 'critic.pth')
        self.critic.load_state_dict(torch.load(policy_weights))
        

    def act(self, curr_obs, mode='eval'):
        if curr_obs==None or not len(self.vision_frames_eval) == 0:
            return self.env_specs['action_space'].sample()
        curr_scent, curr_vision, curr_features, _ = curr_obs
        curr_features = self.pool_features(curr_features)
        curr_vision = self.pool_vision(curr_vision)
        
        self.vision_frames_eval.append(curr_vision)
        self.scent_frames_eval.append(curr_scent)
        self.feature_frames_eval.append(curr_features)
        
        probs = self.policy_function(self._tensor_from_queue(self.feature_frames_eval), self._tensor_from_queue(self.scent_frames_eval),
                                     self._tensor_from_queue(self.vision_frames_eval))
        
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
        self.rewards.append([reward])
        self.actions.append([action])
        self.t += 1
        
        if timestep >= self.num_stack:
            value = self.critic_function(self._tensor_from_queue(self.feature_frames)[5:], 
                                         self._tensor_from_queue(self.scent_frames)[5:],
                                        self._tensor_from_queue(self.vision_frames)[5:])
            probs = self.policy_function(self._tensor_from_queue(self.feature_frames)[:5], 
                                         self._tensor_from_queue(self.scent_frames)[:5],
                                         self._tensor_from_queue(self.vision_frames)[:5])
            dist = torch.distributions.Categorical(probs=probs)
            advantage = self.calculate_return() - value
            #print(self.calculate_return())
            #print(value)
            #print(self.rewards)

            self.critic_loss += advantage.pow(2)
            self.actor_loss += -dist.log_prob(self._tensor_from_queue(self.actions)[5])*advantage.detach()
            
        if self.t == self.step_freq:
            self.optimizer_critic.zero_grad()
            self.critic_loss.backward()
            self.optimizer_critic.step()
            
            self.optimizer_actor.zero_grad()
            self.actor_loss.backward()
            self.optimizer_actor.step()
            
            self.actor_loss = 0
            self.critic_loss = 0
            self.t = 0
            
        if timestep > 4990 and timestep < 5000:
            print(probs)
            print(self.actor_loss)
            print(self.critic_loss)
            
        if done:
            self.model_save()
            

    def calculate_grid():
        pass
    
    def model_save(self):
        PATH = 'policy_lstm.pth'
        torch.save(self.actor_lstm.state_dict(), PATH)
        
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
        inputs = torch.cat([features, scents, vision], dim=1).unsqueeze(1).to(device)

        h, _ = self.actor_lstm(inputs.float())
        #print(h)
        output = self.actor(h[-1])
        #print(output)

        return output
    
    def critic_function(self, features, scents, vision):
        inputs = torch.cat([features, scents, vision], dim=1).unsqueeze(1).to(device)

        output = self.critic(inputs.float())
        
        return torch.squeeze(output).sum()

    def calculate_return(self):
        discounts = np.array([self.discount**i for i in range(int(self.num_stack/2))])
        return np.sum(np.array(self.rewards)[5:].flatten() * discounts)
    
    def _tensor_from_queue(self, queue):
        return torch.tensor(queue).to(device)
    
    def _get_vision(self):
        assert len(self.vision_frames) == self.num_stack, (len(
            self.vision_frames), self.num_stack)
        return LazyFrames(list(self.vision_frames), self.lz4_compress)

    def _get_scent(self):
        assert len(self.scent_frames) == self.num_stack, (len(
            self.scent_frames), self.num_stack)
        return LazyFrames(list(self.scent_frames), self.lz4_compress)

    def _get_feature(self):
        assert len(self.feature_frames) == self.num_stack, (len(
            self.feature_frames), self.num_stack)
        return LazyFrames(list(self.feature_frames), self.lz4_compress)

