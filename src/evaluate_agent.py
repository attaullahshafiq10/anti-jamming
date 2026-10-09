import torch
import numpy as np

def evaluate_tier2_agent(env, policy_net, device):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    actions_taken = []
    total_reward = 0
    energy_used = 0
    
    while True:
        with torch.no_grad():
            q_vals = policy_net(state)
            action = q_vals.argmax(dim=1).item()
            
        next_state_raw, reward, done = env.step(action)
        state = torch.FloatTensor(next_state_raw).unsqueeze(0).to(device)
        
        actions_taken.append(action)
        total_reward += reward
        
        if action == 1:
            energy_used += 5.0
        elif action == 2:
            energy_used += 10.0
        else:
            energy_used += 1.0
            
        if done:
            break
            
    metrics = {
        'total_reward': total_reward,
        'energy_used': energy_used,
        'action_distribution': np.bincount(actions_taken, minlength=3).tolist()
    }
    
    return metrics
