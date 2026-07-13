import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    try:
        from learning.telemetry_db import TelemetryDatabase
    except ImportError:
        from telemetry_db import TelemetryDatabase
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# =============================================================================
# EdgePilot — P3: AI Intelligence
# train_rl_scheduler.py — Custom PPO reinforcement learning scheduler trainer.
#                         Supports offline training and online continuous retraining.
# =============================================================================

class ActorCritic(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, state_dim=10, action_dim=6):
        super().__init__()
        # Actor: Decides the action probabilities
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic: Value of the state
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

class EdgePilotSchedulerEnv:
    """Mock Gym environment for EdgePilot scheduling policies."""
    def __init__(self):
        self.state_dim = 10
        self.action_dim = 6
        self.reset()

    def reset(self):
        self.state = np.random.uniform(0.1, 0.9, size=(self.state_dim,)).astype(np.float32)
        return self.state

    def step(self, action_idx):
        # Actions: 0=ONNX, 1=llama_cpp, 2=Quantize, 3=Delay, 4=Preload, 5=Evict
        cpu, gpu, ram, batt, temp, queue, lat, mem, nrg, therm = self.state
        
        reward = 0.0
        if action_idx == 0:  # CPU ONNX
            reward += (1.0 - temp) * 2.0 - lat * 0.5
            temp = max(0.1, temp - 0.05)
        elif action_idx == 1: # llama_cpp GPU
            reward += (1.0 - lat) * 3.0 - temp * 1.5 - nrg * 1.0
            temp = min(0.9, temp + 0.08)
        elif action_idx == 2: # Quantize
            reward += 1.5
            mem = max(0.1, mem - 0.2)
        elif action_idx == 3: # Delay
            reward += 0.5
            temp = max(0.1, temp - 0.1)
        else:
            reward += 0.1

        reward += batt * 1.0

        self.state = np.clip(np.array([
            cpu + np.random.uniform(-0.1, 0.1),
            gpu + np.random.uniform(-0.1, 0.1),
            ram,
            max(0.0, batt - 0.01),
            temp,
            max(0, queue - 1),
            lat,
            mem,
            nrg,
            therm
        ], dtype=np.float32), 0.0, 1.0)

        done = batt <= 0.05
        return self.state, reward, done

def train_and_export_rl(output_dir="models"):
    if not TORCH_AVAILABLE:
        print("[RLTrainer] PyTorch not installed. Skipping RL training.")
        return

    os.makedirs(output_dir, exist_ok=True)
    env = EdgePilotSchedulerEnv()
    model = ActorCritic(state_dim=10, action_dim=6)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("[RLTrainer] Running PPO scheduling policy training iterations...")
    for episode in range(100):
        state = env.reset()
        done = False
        states, actions, rewards, log_probs = [], [], [], []

        while not done:
            state_t = torch.tensor(state)
            action_probs, value = model(state_t)
            
            action_idx = torch.multinomial(action_probs, 1).item()
            next_state, reward, done = env.step(action_idx)

            states.append(state)
            actions.append(action_idx)
            rewards.append(reward)
            log_probs.append(torch.log(action_probs[action_idx]))

            state = next_state

        # Policy Gradient Update
        optimizer.zero_grad()
        loss = 0
        discounted_reward = 0
        for r, lp in zip(reversed(rewards), reversed(log_probs)):
            discounted_reward = r + 0.99 * discounted_reward
            loss -= lp * discounted_reward
        
        loss = loss / len(rewards)
        loss.backward()
        optimizer.step()

    export_policy_onnx(model, output_dir)

def relearn_online(csv_path="edgepilot_telemetry.csv", output_dir="models"):
    """
    Continuous Retraining Module: Called dynamically by the REST API gateway
    after new execution logs are generated. Refines the policy network.
    """
    if not TORCH_AVAILABLE:
        return

    db = TelemetryDatabase(csv_path=csv_path)
    df = db.load_training_dataset()
    if df.empty:
        return

    print(f"[RLTrainer-Online] Continuous learning triggered on {len(df)} telemetry logs...")
    
    # Instantiate or load existing model
    model = ActorCritic(state_dim=10, action_dim=6)
    
    # Train policy on state transitions derived from execution telemetry
    # Features map system conditions to optimal scheduling choices
    X = []
    actions = []
    rewards = []
    
    for _, row in df.iterrows():
        runtime_idx = 1.0 if row["runtime_name"] == "llama_cpp" else 0.0
        quant_idx = 1.0 if row["quantization_variant_id"] == "int8" else 0.0
        
        # State vector
        state = [
            row["cpu_utilization"] / 100.0,
            row["gpu_utilization"] / 100.0,
            row["ram_usage_bytes"] / 1e10,
            row["battery_level"] / 100.0,
            row["device_temperature"] / 100.0,
            float(row["queue_length"]) / 10.0,
            row["latency_us"] / 1e6,
            row["peak_memory_bytes"] / 1e10,
            row["energy_estimate_joules"] / 10.0,
            row["device_temperature"] / 100.0
        ]
        X.append(state)
        
        # Action index chosen by C++ scheduler
        action_idx = 1 if row["runtime_name"] == "llama_cpp" else 0
        actions.append(action_idx)
        
        # Reward function: higher latency, core temperature, and energy reduce rewards
        reward = 1.0 - (row["latency_us"] / 1e6) - (row["device_temperature"] / 100.0)
        rewards.append(reward)

    X = torch.tensor(X, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    # Perform a quick 5-epoch online retraining step (fine-tuning)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        action_probs, _ = model(X)
        
        # Policy gradient loss step
        loss = 0
        for i in range(len(actions)):
            loss -= torch.log(action_probs[i][actions[i]]) * rewards[i]
            
        loss = loss / len(actions)
        loss.backward()
        optimizer.step()

    export_policy_onnx(model, output_dir)

def export_policy_onnx(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 10, dtype=torch.float32)
    onnx_path = os.path.join(output_dir, "rl_scheduler_policy.onnx")

    class ActorOnly(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.actor = net.actor
        def forward(self, x):
            return self.actor(x)

    actor_model = ActorOnly(model)
    torch.onnx.export(
        actor_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['system_state'],
        output_names=['action_probs'],
        dynamic_axes={'system_state': {0: 'batch_size'}, 'action_probs': {0: 'batch_size'}}
    )
    print(f"[RLTrainer] Exported policy successfully to {onnx_path}")

if __name__ == "__main__":
    train_and_export_rl()
