import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pathlib import Path

def get_epsilon(episode, total_episodes, eps_start, eps_end):
    decay = (eps_start - eps_end) / total_episodes
    eps = eps_start - decay * episode
    return max(eps_end, eps)

def get_beta(episode, total_episodes, beta_start, beta_end):
    inc = (beta_end - beta_start) / total_episodes
    beta = beta_start + inc * episode
    return min(beta_end, beta)

def train_tier2_agent(env, policy_net, target_net, buffer, config, device, save_path):
    optimizer = optim.Adam(policy_net.parameters(), lr=config['lr'])
    criterion = nn.MSELoss(reduction='none')
    
    total_episodes = config['episodes']
    batch_size = config['batch_size']
    gamma = config['gamma']
    
    history = {'rewards': [], 'loss': []}
    
    steps_done = 0
    
    for episode in range(total_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        eps = get_epsilon(episode, total_episodes, config['eps_start'], config['eps_end'])
        beta = get_beta(episode, total_episodes, config['beta_start'], config['beta_end'])
        
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        
        while True:
            # Epsilon Greedy
            if random.random() < eps:
                action = random.randrange(config['action_dim'])
            else:
                with torch.no_grad():
                    q_vals = policy_net(state)
                    action = q_vals.argmax(dim=1).item()
                    
            next_state_raw, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state_raw).unsqueeze(0).to(device)
            
            buffer.push(state.cpu().numpy()[0], action, reward, next_state_raw, done)
            state = next_state
            episode_reward += reward
            steps_done += 1
            
            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size, beta)
                
                states = states.to(device)
                actions = actions.unsqueeze(1).to(device)
                rewards = rewards.unsqueeze(1).to(device)
                next_states = next_states.to(device)
                dones = dones.unsqueeze(1).to(device)
                weights = weights.unsqueeze(1).to(device)
                
                # Double DQN logic
                with torch.no_grad():
                    next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
                    target_q = target_net(next_states).gather(1, next_actions)
                    expected_q = rewards + (gamma * target_q * (1 - dones))
                    
                current_q = policy_net(states).gather(1, actions)
                
                loss = (current_q - expected_q).pow(2) * weights
                prios = loss + 1e-5
                loss = loss.mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                buffer.update_priorities(indices, prios.data.cpu().numpy())
                
                episode_loss += loss.item()
                loss_count += 1
                
            if steps_done % config['target_update'] == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            if done:
                break
                
        history['rewards'].append(episode_reward)
        if loss_count > 0:
            history['loss'].append(episode_loss / loss_count)
            
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_net.state_dict(), save_path)
    
    return history
