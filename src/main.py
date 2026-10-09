import sys
from pathlib import Path
import torch
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs import config
from src.utils import set_seed, ensure_directories
from src.data_loader import load_dataset, clean_dataset, separate_features
from src.preprocessing import run_preprocessing_pipeline
from src.dataset import get_dataloaders
from src.model import HybridPerceptionModel
from src.dqn_model import DuelingDoubleDQN
from src.train_perception import train_tier1_model
from src.evaluate_perception import evaluate_tier1_model
from src.environment import WirelessEnvironment
from src.replay_buffer import PrioritizedReplayBuffer
from src.train_agent import train_tier2_agent
from src.evaluate_agent import evaluate_tier2_agent
from src.metrics import calculate_model_complexity, estimate_mflops
from src.visualization import plot_confusion_matrix, plot_training_curves, plot_reward_convergence

def main():
    set_seed(config.RANDOM_SEED)
    ensure_directories([config.MODELS_DIR, config.RESULTS_DIR, config.LOGS_DIR])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Loading
    print("Loading data...")
    # Use preprocessed data if it exists, otherwise fallback to raw data
    if config.PREPROCESSED_DATA_FILE.exists():
        data_path = config.PREPROCESSED_DATA_FILE
    elif config.RAW_DATA_FILE.exists():
        data_path = config.RAW_DATA_FILE
    else:
        print("Dataset files not found in the data directory. Please provide data/preprocess_data.csv or data/raw_data.csv.")
        return
        
    df = load_dataset(data_path)
    
    # Identify target column dynamically or use a known one
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    
    df = clean_dataset(df, target_col)
    
    # 2. Preprocessing
    print("Preprocessing...")
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler, le = run_preprocessing_pipeline(
        df, target_col, config.SEQUENCE_LENGTH, config.TRAIN_TEST_SPLIT
    )
    
    num_features = X_train_seq.shape[2]
    
    train_loader, test_loader = get_dataloaders(
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, config.TIER1_BATCH_SIZE
    )
    
    # 3. Tier 1 Model Training
    print("Initializing Tier 1 Model...")
    perception_model = HybridPerceptionModel(
        num_features=num_features,
        seq_length=config.SEQUENCE_LENGTH,
        num_classes=config.NUM_CLASSES,
        cnn_f1=config.TIER1_CNN_FILTERS_1,
        cnn_f2=config.TIER1_CNN_FILTERS_2,
        lstm_hidden=config.TIER1_BILSTM_HIDDEN
    )
    
    print("Training Tier 1 Perception Module...")
    tier1_save_path = config.MODELS_DIR / "tier1_model.pth"
    history = train_tier1_model(
        perception_model, train_loader, test_loader, 
        config.TIER1_EPOCHS, config.TIER1_LEARNING_RATE, device, tier1_save_path
    )
    
    plot_training_curves(history, config.RESULTS_DIR / "tier1_training_curves.png")
    
    # 4. Tier 1 Evaluation
    print("Evaluating Tier 1 Perception Module...")
    perception_model.load_state_dict(torch.load(tier1_save_path))
    metrics, cm, probs, preds, labels = evaluate_tier1_model(
        perception_model, test_loader, device, config.NUM_CLASSES
    )
    
    class_names = [config.CLASS_MAPPING[i] for i in range(config.NUM_CLASSES)]
    plot_confusion_matrix(cm, class_names, config.RESULTS_DIR / "tier1_confusion_matrix.png")
    
    params, size_mb = calculate_model_complexity(perception_model)
    mflops = estimate_mflops(perception_model, (1, config.SEQUENCE_LENGTH, num_features))
    
    print(f"Tier 1 Metrics: Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    print(f"Tier 1 Complexity: Parameters: {params}, Size: {size_mb:.2f} MB, MFLOPs: {mflops:.2f}")
    
    # 5. Tier 2 Environment Setup
    print("Setting up Tier 2 Environment...")
    # Generate flattened state for Tier 2 for simplicity
    flat_data = df.drop(columns=[target_col]).values
    
    reward_weights = {
        'alpha': config.REWARD_ALPHA,
        'beta': config.REWARD_BETA,
        'delta': config.REWARD_DELTA,
        'lambda_': config.REWARD_LAMBDA,
        'mu': config.REWARD_MU
    }
    
    env = WirelessEnvironment(flat_data, reward_weights)
    state_dim = flat_data.shape[1]
    action_dim = 3
    
    # 6. Tier 2 RL Agent Training
    print("Initializing Tier 2 Agent...")
    policy_net = DuelingDoubleDQN(state_dim, action_dim).to(device)
    target_net = DuelingDoubleDQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    buffer = PrioritizedReplayBuffer(config.TIER2_REPLAY_CAPACITY, config.TIER2_PER_ALPHA)
    
    rl_config = {
        'lr': config.TIER2_LEARNING_RATE,
        'episodes': 300,
        'batch_size': 64,
        'gamma': config.TIER2_GAMMA,
        'target_update': config.TIER2_TARGET_UPDATE,
        'action_dim': action_dim,
        'eps_start': config.TIER2_EPSILON_START,
        'eps_end': config.TIER2_EPSILON_END,
        'beta_start': config.TIER2_PER_BETA_START,
        'beta_end': config.TIER2_PER_BETA_END
    }
    
    print("Training Tier 2 Mitigation Engine...")
    tier2_save_path = config.MODELS_DIR / "tier2_model.pth"
    rl_history = train_tier2_agent(env, policy_net, target_net, buffer, rl_config, device, tier2_save_path)
    
    plot_reward_convergence(rl_history['rewards'], config.RESULTS_DIR / "tier2_reward_convergence.png")
    
    # 7. Tier 2 Evaluation
    print("Evaluating Tier 2 Mitigation Engine...")
    policy_net.load_state_dict(torch.load(tier2_save_path))
    rl_metrics = evaluate_tier2_agent(env, policy_net, device)
    
    print(f"Tier 2 Evaluation: Total Reward: {rl_metrics['total_reward']:.2f}, Energy Used: {rl_metrics['energy_used']:.2f}")
    print(f"Action Distribution: {rl_metrics['action_distribution']}")
    
    print("Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
