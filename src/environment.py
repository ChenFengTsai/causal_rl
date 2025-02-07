import gymnasium
import numpy as np
import torch
from sklearn.decomposition import PCA
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModifiedEnv(gymnasium.Env):
    def __init__(self, original_env, dynamics_model, policy, extra_feature_dim, pca_update_interval=5000, reduce_feature=True):
        """
        Custom Gym environment wrapper.

        :param original_env: The original Gym environment.
        :param dynamics_model: The dynamics model predicting the next state.
        :param policy: The policy model (for prediction).
        :param extra_feature_dim: Number of extra features to keep if PCA is applied.
        :param pca_update_interval: How often to update PCA (every X steps).
        :param reduce_feature: Boolean flag to enable/disable PCA-based feature reduction.
        """
        self.original_env = original_env
        self.dynamics_model = dynamics_model
        self.policy = policy  
        self.extra_feature_dim = extra_feature_dim

        self.pca_update_interval = pca_update_interval
        self.reduce_feature = reduce_feature  # Toggle feature reduction

        # Initialize PCA and buffer to store states for dynamic updates
        self.pca = None
        self.state_buffer = deque(maxlen=1000)  # Store the last 1000 states for PCA training
        self.steps_since_last_pca_update = 0  # Track steps since last PCA update

        original_obs_space = original_env.observation_space
        self.observation_space = gymnasium.spaces.Box(
            low=np.hstack((original_obs_space.low, [-np.inf] * self.extra_feature_dim)),
            high=np.hstack((original_obs_space.high, [np.inf] * self.extra_feature_dim)),
            dtype=np.float32
        )
        self.action_space = original_env.action_space
        
    def reset(self, seed=None, options=None):
        obs, info = self.original_env.reset(seed=seed, options=options)
        peek_feature = self._generate_extra_feature(obs, np.zeros(self.action_space.shape))

        peek_feature = peek_feature.detach().cpu().numpy()
        if not self.reduce_feature:
            peek_feature = peek_feature[0]

        combined_obs = np.hstack((obs, peek_feature))
        self.current_obs = obs

        # # Debugging Prints
        # print("=== RESET ===")
        # print("Obs shape:", obs.shape)
        # print("Extra feature shape:", peek_feature.shape)
        # print("Combined obs shape (reset):", combined_obs.shape)

        return combined_obs, info


    def step(self, action):
        next_obs, reward, done, truncated, info = self.original_env.step(action)

        peek_feature = self._generate_extra_feature(self.current_obs, action)
        peek_feature = peek_feature.detach().cpu().numpy()
        if not self.reduce_feature:
            peek_feature = peek_feature[0]

        augmented_obs = np.hstack((self.current_obs, peek_feature))

        predicted_action, _ = self.policy.predict(augmented_obs, deterministic=True)

        next_peek_feature = self._generate_extra_feature(next_obs, predicted_action)
        next_peek_feature = next_peek_feature.detach().cpu().numpy()
        if not self.reduce_feature:
            next_peek_feature = next_peek_feature[0]

        combined_obs = np.hstack((next_obs, next_peek_feature))

        self.current_obs = next_obs

        return combined_obs, reward, done, truncated, info

    def update_pca(self):
        """Dynamically update PCA using stored state samples."""
        if self.reduce_feature and len(self.state_buffer) >= 100:  # Ensure enough samples
            state_samples = np.array(self.state_buffer)
            self.pca = PCA(n_components=self.extra_feature_dim)
            self.pca.fit(state_samples)
            print(f"[PCA Updated] Trained on {len(self.state_buffer)} samples.")

    def _generate_extra_feature(self, obs, action):
        """
        Generate extra features based on the dynamics model's prediction.

        :param obs: The current observation.
        :param action: The action taken by the agent.
        :return: Augmented observation with extra features.
        """
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict next state using the dynamics model
        next_state_pred = self.dynamics_model(state_tensor, action_tensor)
        next_state_pred_cpu = next_state_pred.detach().cpu().numpy()[0]

        # Store states for PCA updates
        self.state_buffer.append(next_state_pred_cpu)
        self.steps_since_last_pca_update += 1

        # Update PCA periodically
        if self.steps_since_last_pca_update >= self.pca_update_interval:
            self.update_pca()
            self.steps_since_last_pca_update = 0

        # Apply PCA-based feature reduction if enabled
        if self.reduce_feature:
            if self.pca is not None and len(self.state_buffer) >= 100:
                reduced_features = self.pca.transform(next_state_pred_cpu.reshape(1, -1))[0]
            else:
                # Fallback: Use the first `extra_feature_dim` features if PCA is not ready
                # feature_indices = np.argsort(np.var(next_state_pred_cpu, axis=0))[-self.extra_feature_dim:]
                # reduced_features = next_state_pred_cpu[feature_indices]
                reduced_features = next_state_pred_cpu[:self.extra_feature_dim]

            # Convert back to PyTorch tensor and move to GPU
            reduced_features_tensor = torch.tensor(reduced_features, dtype=torch.float32).to(device)

            return reduced_features_tensor
        else:
            # No feature reduction → Return full next state prediction
            return next_state_pred

