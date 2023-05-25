import gym
from helpers import NormalizedEnv, RandomAgent
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from mpl_toolkits.mplot3d import Axes3D
from pylab import*
import os
from tqdm import tqdm
import random as rand

# Implement a heuristic policy
class HeuristicPendulumAgent():
    def __init__(self, env):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.agent_name='HeuristicPendulumAgent'

    def get_agent_name(self):
        return self.agent_name
        

    def compute_action(self, state,fix_torque=1):
        #  print('fix_torque',fix_torque)
         # When the pendulum is in the lower half of the domain (x<0)
         if state[0]<0:
             # applies a fixed torque in the same direction as the pendulum’s angular velocity
             return fix_torque*np.sign(state[2])
         # When the pendulum is in the higher half of the domain (x>0)
         else:
             # applies a fixed torque in the  opposite direction as the pendulum’s angular velocity
             return -1*fix_torque*np.sign(state[2])

# Implement the Replay Buffer
class ReplayBuffer():
    def __init__(self, max_size):
        # amount: how many transitions are stored in the replay buffer now
        self.amount=0
        self.full_buffer=False
        # max_size: how many transitions replay buffer can store at most
        self.max_size = max_size
        # self.total_transition = []
        self.total_transition = np.zeros((max_size, 9))

    def add(self, transition):
        # transition: a tuple of (state, action, reward, next_state, trunc)
        if self.amount == self.max_size:
             # check if the replay buffer is full
            self.full_buffer=True
            # print('self.full_buffer:',self.full_buffer)
            self.amount=0

        if self.full_buffer:
            # print("The replay buffer is full.-> remove the oldest transition")
            # remove the oldest transition
            # print('self.total_transition[self.amount]:', self.total_transition[self.amount])
            self.total_transition[self.amount]=np.array(transition)
            # print('self.total_transition[self.amount]:', self.total_transition[self.amount])
        else:
            # add transition to replay buffer
            # print('self.total_transition[self.amount-1]:',self.total_transition[self.amount-1])
            # print('transition:',transition)
            # print('transition:',np.array(transition))
            self.total_transition[self.amount] = np.array(transition)
        self.amount+=1
        
        
    def sample(self, batch_size):
        # batch_size: how many transitions will be sampled
        # return a batch of transitions
        
        # check if the replay buffer is empty
        if len(self.total_transition) == 0:
            print("The replay buffer is empty.")
            return None

        # check if the replay buffer has enough transitions
        if len(self.total_transition) < batch_size:
            print("The replay buffer does not have enough transitions.")
            return None
        
        # sample batch_size transitions from the replay buffer randomly
        # return random.sample(self.total_transition, batch_size)
        random_index = rand.sample(range(0, self.amount), batch_size)
        # assert np.sum(np.array(random_index)>self.amount)==0, "random_index should be less than 'self.amount'"
        # print('amount:',self.amount)
        # print('random_index:',random_index)
        return np.array(self.total_transition[random_index])
    def __len__(self):
        return self.amount

# Implement the class QNetwork
class QNetwork(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, hidden_dim=32):
        super(QNetwork, self).__init__()
        # input_dim: dimension of the input -> 3 elements of the state and 1 of the action
        # hidden_dim: dimension of one hidden layer -> 32 nodes
        # output_dim: dimension of action -> a scalar value (the expected cumulative reward)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, state):
        # state: state -> 3 elements of the state and 1 of the action
        # return expected cumulative reward 
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x   
    
# Implement the 1-step TD-learning rule for the QNetwork
def train_QNetwork(Q_network,Q_target,agent, batch, gamma, optimizer,fix_torque=1,policy_network=None,deterministic=None,policy_target= None):
    # print("HERE TRAIN")
    # Q_network: Q Network
    # agent: policy use for action selection
    # batch: a batch of transitions
    # gamma: discount factor
    # optimizer: optimizer
    # fix_torque: fixed torque for HeuristicPendulumAgent
    
    # unpack the batch of transitions
    # input_batch:put together 3 elements of the state and 1 of the action
    input_batch=torch.tensor(np.concatenate((batch[:,0:3],np.expand_dims(batch[:,3], axis=1)),axis=1), dtype=torch.float)
    reward_batch = torch.tensor(batch[:,4], dtype=torch.float)
    reward_batch=torch.unsqueeze(reward_batch, 1)
    if agent.agent_name =='HeuristicPendulumAgent':
        next_action=np.array([agent.compute_action(transition[5:8],fix_torque) for transition in batch])
        target_batch=torch.tensor(np.concatenate((batch[:,5:8],np.expand_dims((next_action),axis=1)),axis=1), dtype=torch.float)
    elif agent.agent_name =='DDPGAgentSoftUpdate':
        target_batch=torch.tensor(np.concatenate((batch[:,5:8],agent.compute_action(policy_target,batch[:,5:8],deterministic)),axis=1), dtype=torch.float)
    else:
        target_batch=torch.tensor(np.concatenate((batch[:,5:8],agent.compute_action(policy_network,batch[:,5:8],deterministic)),axis=1), dtype=torch.float)
    trunc_batch=torch.tensor(batch[:,8], dtype=torch.float)

    # clear the gradients of the optimizer
    optimizer.zero_grad()

    # compute the Q values of the current state + action pair
    # Q_network.train()
    Q_values = Q_network(input_batch)
    
    # The target should not be differentiated, i.e., wrap it in a with torch.no grad()
    with torch.no_grad():
        # Q_network.eval()
        # compute the Q values of the next state + next action pair
        Q_next_values = Q_target(target_batch)
    Q_target_values = reward_batch + gamma * Q_next_values * torch.unsqueeze((1 - trunc_batch), 1)
        
    # compute the loss
    loss = F.mse_loss(Q_values.view(-1, 1), Q_target_values.view(-1, 1))
    # print('loss:',loss.requires_grad)
    
    # compute the gradients of the loss
    loss.backward()
    # # show the gradients of the parameters in the Q network
    # print("Q_network:",Q_network.fc1.weight.grad[0])
    # try:
    #     print("Q_target:",Q_target.fc1.weight.grad[0])
    # except:
    #     pass
    
    # update the weights of the Q network
    optimizer.step()
    
    return loss.item(), Q_network

# PolicyNetwork
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_dim=32):
        super(PolicyNetwork, self).__init__()
        # input_dim: dimension of the input -> 3 elements of the state
        # hidden_dim: dimension of one hidden layer -> 32 nodes
        # output_dim: dimension of action -> a scalar (action)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, state):
        # state: state -> 3 elements of the state
        # return action 
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # apply tanh to restrict the action in the range [-1, 1]
        x = F.tanh(self.fc3(x))

        return x

# Train PolicyNetwork
def train_PolicyNetwork(policy_network,Q_network, batch, optimizer):
    # print("HERE TRAIN")
    # optimizer: optimizer
    
    # unpack the batch of transitions
    state_batch=torch.tensor(batch[:,0:3], dtype=torch.float)

    # clear the gradients of the optimizer
    optimizer.zero_grad()

    
    # policy_network: policy use for action selection
    predicted_action = policy_network(state_batch)
    # with torch.no_grad():
    # compute the Q values of the current state + predicted action pair
    Q_values = Q_network(torch.cat((state_batch,predicted_action),1))
        
    # print("Q_values:",Q_values.mean())
    # print(' -Q_values.mean():', -Q_values.mean())
    # compute the loss
    loss = -Q_values.mean()
    # print('loss:',loss)
    
    
    # compute the gradients of the loss
    loss.backward()
    
    # update the weights of the Policy network
    optimizer.step()
    # print('loss.item():',loss.item())
    return loss.item(),policy_network
    
#GaussianActionNoise
class GaussianActionNoise():
    def __init__(self, sigma):
        self.sigma = sigma
        self.mu=0
    def get_noisy_action(self,action):
        noise = np.random.normal(self.mu,self.sigma)
        noisy_action = action + noise
        # clip the noisy action to be in the range [-1, 1]
        noisy_action = np.clip(noisy_action, -1, 1)
        return noisy_action
    
#OUActionNoise
class OUActionNoise():
    def __init__(self, sigma, theta):
        self.sigma= sigma
        self.mu = 0
        self.theta = theta
        self.noise = 0
    
        
    def get_noisy_action(self,action):
        self.evolve_state()
        noisy_action = action + self.noise
        # clip the noisy action to be in the range [-1, 1]
        noisy_action = np.clip(noisy_action, -1, 1)
        return noisy_action

    def evolve_state(self):
        standard_normal_distribution= np.random.normal(0,1)
        self.noise = (1-self.theta)*self.noise + self.theta*standard_normal_distribution

# DDPGAgent with soft update
class DDPGAgent():
    def __init__(self,env,sigma,softUpdate=False,tau=0,theta=0, action_noise='gaussian'):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.sigma=sigma
        self.theta=theta
        if softUpdate:
            self.agent_name='DDPGAgentSoftUpdate'
            self.tau = tau
        else:
            self.agent_name='DDPGAgent'
        if action_noise == 'gaussian':
            self.action_noise = GaussianActionNoise(sigma)
        else:
            self.action_noise = OUActionNoise(sigma,theta)

    def compute_action(self,actor_network,state,deterministic=True):
        # deterministic: regulates whether to add random noise to the action or not
        with torch.no_grad():
            # convert state to tensor
            state = torch.tensor(state, dtype=torch.float)
            # compute the action
            action = actor_network(state)
            if not deterministic:
                # add random noise to the action
                action = self.action_noise.get_noisy_action(action)
            # convert action to numpy array
            action = action.numpy()
        return action
    
    def update_target_params(self,actor_network,critic_network,actor_target,critic_target):
        # update the parameters of networks
        with torch.no_grad():
            for param, target_param in zip(critic_network.parameters(), critic_target.parameters()):
                    target_param.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(actor_network.parameters(), actor_target.parameters()):
                target_param.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return critic_target, actor_target
    
    def reset(self):
        self.action_noise = OUActionNoise(self.sigma,self.theta)


def seven_different_theta():
    # training the policies using one Tau = 0.01, theta between 0 and 1
    env = gym.make("Pendulum-v1")
    num_episode=1000
    num_step=200
    batch_size=128
    buffer_size=100000
    gamma=0.99
    learning_rate=1e-4
    sigma=0.3
    tau =0.01
    deterministic=False
    theta_list=np.linspace(0.1,1,5)
    for theta in theta_list:    
        list_average_loss_critic_network=[]
        list_average_loss_actor_network=[]
        list_reward=[]
        # initialize the critic network and the critic target 
        critic_network=QNetwork()
        critic_target=QNetwork()
        critic_target.load_state_dict(critic_network.state_dict())
        # define actor network and the actor target
        actor_network=PolicyNetwork()
        actor_target=PolicyNetwork()
        actor_target.load_state_dict(actor_network.state_dict())
        # define a optimizer for the QNetwork
        critic_optimizer = torch.optim.Adam(critic_network.parameters(), learning_rate)
        actor_optimizer = torch.optim.Adam(actor_network.parameters(), learning_rate)
        # define replay buffer
        replay_buffer=ReplayBuffer(buffer_size)
        # define agent
        agent=DDPGAgent(env,sigma,softUpdate=True,tau=tau,theta=theta,action_noise='OUActionNoise')
        print('Start training...')
        for episode in tqdm(range(num_episode)):
            agent.reset()
            sum_reward=0
            # print('episode number:',episode,'/',num_episode,'...')
            # initialize the sum_loss for each episode
            sum_loss_critic_network=0
            sum_loss_actor_network=0
            # renew the environment for every episode
            state, info = env.reset()
            normalized_env = NormalizedEnv(env)
            
            for step in range(num_step):
                # print('step:',step,'/',num_step,'...')
                
                # add transition to replay buffer, and kick out the old one if the buffer is full
                # add noise to action when deterministic=False
                action = agent.compute_action(actor_network,state,deterministic)
                normalized_action = normalized_env.action(action)
                next_state, reward, terminated, truncated, info = env.step(normalized_action)
                transition=np.concatenate((state, action, reward, next_state, truncated),axis=None)
                replay_buffer.add(transition)
                state=next_state.copy()
                # cumulative reward
                sum_reward+=reward
                if replay_buffer.__len__() < batch_size:
                    print("The replay buffer does not have enough transitions.")
                    continue
            
                # sample a batch of transitions from the replay buffer
                batch = replay_buffer.sample(batch_size)

                # train the policy_network
                loss_actor_network,actor_network=train_PolicyNetwork(actor_network,critic_network, batch, actor_optimizer)

                loss_critic_network,critic_network=train_QNetwork(critic_network,critic_target,agent, batch, gamma, critic_optimizer,fix_torque=None,policy_network=actor_network,deterministic=deterministic,policy_target=actor_target)
                if loss_critic_network is not None:
                    sum_loss_critic_network+=loss_critic_network
                if loss_actor_network is not None:   
                    sum_loss_actor_network+=loss_actor_network

                critic_target, actor_target=agent.update_target_params(actor_network,critic_network,actor_target,critic_target)
                
            average_loss_critic_network=sum_loss_critic_network/num_step
            average_loss_actor_network=sum_loss_actor_network/num_step
            list_average_loss_critic_network.append(average_loss_critic_network)
            list_average_loss_actor_network.append(average_loss_actor_network)
            list_reward.append(sum_reward)

        print('Finishing training!')
        print('Save the model...')
        # save the model with path name with real-time date and time
        save_path_critic_OUActionNoise= os.path.join('trained_model','7_CriticNetwork_DDPG_OUActionNoise_'+datetime.datetime.now().strftime("%Y%m%d%H%M")+'_ep'+str(num_episode)+'_tau'+str(tau)+'_theta'+str(theta)+'.pth')
        save_path_critic_target_OUActionNoise= os.path.join('trained_model','7_CriticTarget_DDPG_OUActionNoise_'+datetime.datetime.now().strftime("%Y%m%d%H%M")+'_ep'+str(num_episode)+'_tau'+str(tau)+'_theta'+str(theta)+'.pth')
        torch.save(critic_network.state_dict(), save_path_critic_OUActionNoise)
        torch.save(critic_target.state_dict(),save_path_critic_target_OUActionNoise)
        print('critic_network Model save to ',save_path_critic_OUActionNoise,' !')
        print('critic_target Model save to ',save_path_critic_target_OUActionNoise,' !')
        save_path_actor_OUActionNoise= os.path.join('trained_model','7_ActorNework_DDPG_OUActionNoise_'+datetime.datetime.now().strftime("%Y%m%d%H%M")+'_ep'+str(num_episode)+'_tau'+str(tau)+'_theta'+str(theta)+'.pth')
        save_path_actor_target_OUActionNoise= os.path.join('trained_model','7_ActorTarget_DDPG_OUActionNoise_'+datetime.datetime.now().strftime("%Y%m%d%H%M")+'_ep'+str(num_episode)+'_tau'+str(tau)+'_theta'+str(theta)+'.pth')
        torch.save(actor_network.state_dict(), save_path_actor_OUActionNoise)
        torch.save(actor_target.state_dict(), save_path_actor_target_OUActionNoise)
        print('actor_network Model save to ',save_path_actor_OUActionNoise,' !')
        print('actor_target Model save to ',save_path_actor_target_OUActionNoise,' !')

        list_number_episodes = np.linspace(1,num_episode,num = num_episode)
        plt.figure()
        plt.plot(list_number_episodes,list_average_loss_critic_network, label='Critic Network using OUActionNoise')
        plt.plot(list_number_episodes,list_average_loss_actor_network, label='Actor Network using OUActionNoise')
        plt.xlabel('Number of episodes')
        plt.ylabel('Average loss per episode')
        plt.title(f'Average loss obtained by the critic Network and the actor Network using OUActionNoise, tau = {tau} , theta = {theta}')
        plt.legend()
        print('Save the figures...')
        image_path=os.path.join('figure','7_AverageLoss_OUActionNoise_'+datetime.datetime.now().strftime("%Y%m%d%H%M")+'_ep'+str(num_episode)+'_tau'+str(tau)+'_theta'+str(theta)+'.png')
        plt.savefig(image_path)
        # plt.show()

        plt.figure()
        plt.plot(list_number_episodes,list_reward)
        plt.xlabel('Number of episodes')
        plt.ylabel('Accumulated reward using OUActionNoise')
        plt.title(f"Accumulated reward per episode using OUActionNoise, tau = {tau}, theta = {theta}")
        print('Save the figures...')
        image_path=os.path.join('figure','7_AccumulatedRewardPerEpisode__OUActionNoise_'+datetime.datetime.now().strftime("%Y%m%d%H%M")+'_ep'+str(num_episode)+'_tau'+str(tau)+'_theta'+str(theta)+'.png')
        plt.savefig(image_path)
        # plt.show()
    env.close()

if __name__ == "__main__":
    seven_different_theta()