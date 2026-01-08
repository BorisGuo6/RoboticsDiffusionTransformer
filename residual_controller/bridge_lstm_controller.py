import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from visual_encoder import DINOv2Encoder
from bridge.bridge_model import StochasticInterpolants
from controller_dataset import denormalize_actions, normalize_actions


class BridgeLSTMController:
    """
    Combined Diffusion-LSTM controller that refines VLA actions using:
    1. A stochastic interpolants model to generate initial refined action trajectories
    2. An LSTM model to adjust these trajectories based on real-time force feedback

    This architecture allows for both offline planning (diffusion) and online adaptation (LSTM).
    """

    def __init__(
            self,
            state_dim=10,
            hidden_dim=256,
            lstm_hidden_dim=128,
            image_model_path="facebook/dinov2-small",
            diffusion_steps=10,
            device="cuda",
            model_args=None,
            force_dim=3,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.device = device
        self.diffusion_steps = diffusion_steps
        self.force_dim = force_dim
        self.model_args = model_args
        self.stats = None

        # Initialize image encoder
        self.image_encoder = DINOv2Encoder(model_name=image_model_path, device=device)
        self.latent_obs_dim = self.image_encoder.hidden_size

        # Define observation dimension (includes images and state)
        self.obs_dim = self.latent_obs_dim * 2 + self.state_dim + self.force_dim

        # Initialize state encoder (for diffusion model conditioning)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Initialize LSTM model for online adaptation
        self.lstm_input_dim = self.state_dim + self.force_dim  # Action + encoded force
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Initialize action decoder (LSTM output -> refined action)
        self.action_decoder = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.state_dim)
        )

        # Collect all trainable models in a ModuleList for easy management
        self.models = nn.ModuleList([
            self.state_encoder,
            self.lstm,
            self.action_decoder
        ])

        # Initialize diffusion model
        self.diffusion_model = StochasticInterpolants()
        if self.model_args:
            self.diffusion_model.load_model(self.model_args, device)

        # Move all models to device
        self.to(device)

    def to(self, device):
        """Move all models to the specified device."""
        self.device = device
        # Using ModuleList makes this very simple
        self.models.to(device)
        return self

    def encode_images(self, images_cam1, images_cam2):
        """
        Process images from both cameras through the DINOv2 encoder.

        Args:
            images_cam1: Front camera images [batch_size, H, W, C]
            images_cam2: Right camera images [batch_size, H, W, C]

        Returns:
            cam1_features, cam2_features: Encoded image features
        """
        if images_cam1 is None or images_cam2 is None:
            return None, None

        # Ensure images are on the correct device
        images_cam1 = images_cam1.to(self.device)
        images_cam2 = images_cam2.to(self.device)

        # Extract features using DINOv2
        cam1_features = self.image_encoder.forward(images_cam1)
        cam2_features = self.image_encoder.forward(images_cam2)

        return cam1_features, cam2_features

    def encode_observation(self, state, images_cam1=None, images_cam2=None, forces=None):
        """
        Encode state, image observations, and forces into a conditioning vector.

        Args:
            state: Robot state [batch_size, state_dim]
            images_cam1: Optional front camera image [batch_size, H, W, C]
            images_cam2: Optional right camera image [batch_size, H, W, C]
            forces: Force measurements [batch_size, force_dim]

        Returns:
            obs_cond: Encoded observation for conditioning [batch_size, hidden_dim]
        """
        # Ensure state is on the correct device
        state = state.to(self.device)
        forces = forces.to(self.device) if forces is not None else torch.zeros(state.shape[0], self.force_dim,
                                                                               device=self.device)

        # Encode images
        cam1_features, cam2_features = self.encode_images(images_cam1, images_cam2)

        # Concatenate state with forces for full observation
        full_state = torch.cat((state, forces), dim=-1)

        # Encode the full observation
        obs_cond = self.state_encoder(torch.cat((cam1_features, cam2_features, full_state), dim=-1))

        return obs_cond


    def get_lstm_loss(self, batch):

        obs_cond = batch['obs_cond'].to(self.device)
        vla_actions_n = batch['vla_act'].to(self.device)
        expert_actions_n = batch['expert_act'].to(self.device)
        force = batch['forces'].to(self.device)
        steps = self.diffusion_steps

        with torch.no_grad():
            # Get refined action trajectory from diffusion model
            diff_actions_n = self.diffusion_model.sample(
                x_prior=vla_actions_n,
                cond=obs_cond,
                diffuse_step=steps
            )

        # Initialize LSTM hidden state
        batch_size = diff_actions_n.shape[0]
        h0 = torch.zeros(2, batch_size, self.lstm_hidden_dim, device=self.device)
        c0 = torch.zeros(2, batch_size, self.lstm_hidden_dim, device=self.device)
        lstm_hidden = (h0, c0)

        # Get horizon (sequence length)
        horizon = diff_actions_n.shape[1]
        # Initialize container for LSTM processed actions
        lstm_delta_actions = torch.zeros_like(diff_actions_n).to(self.device)

        # Process action sequence step by step with force feedback
        for t in range(horizon):
            # Process the current step through LSTM with force feedback
            tac_delta_action, lstm_hidden = self.predict_step(
                diffusion_action=diff_actions_n[:, t],
                force=force[:, t],
                lstm_hidden=lstm_hidden
            )

            # Store the refined action
            lstm_delta_actions[:, t] = tac_delta_action

        refined_actions_n = diff_actions_n + lstm_delta_actions

        return F.mse_loss(expert_actions_n, refined_actions_n, reduction='mean')

    def predict_sequence(self, state, vla_actions, images_cam1=None, images_cam2=None,
                         force=None):
        """
        Generate a complete refined action sequence by:
        1. Using diffusion model to generate initial refined action trajectory
        2. Processing each step with LSTM for online adaptation based on force feedback

        Args:
            state: Current robot state [batch_size, state_dim]
            vla_actions: VLA actions to refine [batch_size, horizon, state_dim]
            images_cam1: Optional front camera image [batch_size, H, W, C]
            images_cam2: Optional right camera image [batch_size, H, W, C]
            initial_force: Initial force measurement [batch_size, force_dim]
            diffuse_steps: Number of diffusion steps (default: self.diffusion_steps)

        Returns:
            lstm_actions: Final LSTM-refined action sequence [batch_size, horizon, state_dim]
        """
        # Set models to evaluation mode
        self.eval()
        init_force = force[:,0,:]
        # Get conditional encoding
        obs_cond = self.encode_observation(state, images_cam1, images_cam2, init_force)

        # Normalize VLA actions
        vla_actions_n = normalize_actions(vla_actions, self.stats, 'vla')

        # Generate initial refined action sequence using diffusion
        with torch.no_grad():
            # Get refined action trajectory from diffusion model
            diff_actions_n = self.diffusion_model.sample(
                x_prior=vla_actions_n,
                cond=obs_cond,
                diffuse_step=self.diffusion_steps
            )

            # Initialize LSTM hidden state
            batch_size = state.shape[0]
            h0 = torch.zeros(2, batch_size, self.lstm_hidden_dim, device=self.device)
            c0 = torch.zeros(2, batch_size, self.lstm_hidden_dim, device=self.device)
            lstm_hidden = (h0, c0)

            # Get horizon (sequence length)
            horizon = diff_actions_n.shape[1]
            # Initialize container for LSTM processed actions
            lstm_delta_actions = torch.zeros_like(diff_actions_n)

            # Process action sequence step by step with force feedback
            for t in range(horizon):

                # Process the current step through LSTM with force feedback
                tac_delta_action, lstm_hidden = self.predict_step(
                    diffusion_action=diff_actions_n[:, t],
                    force=force[:,t],
                    lstm_hidden=lstm_hidden
                )

                # Store the refined action
                lstm_delta_actions[:, t] = tac_delta_action

            refined_actions_n = diff_actions_n + lstm_delta_actions
            # Denormalize actions
            refined_lstm_actions = denormalize_actions(refined_actions_n, self.stats, 'expert')
            return refined_lstm_actions

    def predict_step(self, diffusion_action, force, lstm_hidden):
        """
        Refine a single action step using the LSTM with force feedback.

        The key idea is that we use both:
        1. The action from the diffusion model for this step
        2. The force measurement from the environment
        3. The previously executed action

        Args:
            diffusion_action: Action from diffusion model for this step [batch_size, state_dim]
            current_action: Most recently executed action [batch_size, state_dim]
            force: Current force measurement [batch_size, force_dim]
            lstm_hidden: Current LSTM hidden state (h, c)

        Returns:
            refined_action: LSTM-refined action for this step [batch_size, state_dim]
            next_lstm_hidden: Updated LSTM hidden state for next step
        """
        self.eval()
        with torch.no_grad():

            # Combine diffusion action and encoded force as LSTM input
            # Using the diffusion action from the trajectory as a reference plan
            lstm_input = torch.cat([diffusion_action, force], dim=-1).unsqueeze(1)

            # Process through LSTM
            lstm_output, next_lstm_hidden = self.lstm(lstm_input, lstm_hidden)

            # Decode LSTM output to get action adjustment
            tac_delta_action = self.action_decoder(lstm_output.squeeze(1))

            return tac_delta_action, next_lstm_hidden

    def train(self):
        """Set models to training mode."""
        self.models.train()
        self.diffusion_model.train()
        return self

    def eval(self):
        """Set models to evaluation mode."""
        self.models.eval()
        self.diffusion_model.eval()
        return self

    def save(self, path):
        """
        Save the controller model.

        Args:
            path: Path to save the model
        """
        state_dict = {
            'models': self.models.state_dict(),
            'model_args': self.model_args,
            'stats': self.stats
        }

        # Save controller components
        torch.save(state_dict, f"{path}/controller.pt")

        # Save diffusion model separately
        self.diffusion_model.save_model(path)

    def load(self, path):
        """
        Load the controller model.

        Args:
            path: Path to load the model from
        """
        # Load controller components
        checkpoint = torch.load(f"{path}/controller.pt", map_location=self.device)

        # Load the entire ModuleList at once
        self.models.load_state_dict(checkpoint['models'])

        self.model_args = checkpoint['model_args']
        self.stats = {key: torch.tensor(value, dtype=torch.float32).cuda() for key, value in
                      checkpoint['stats'].items()}

        # Load diffusion model
        self.diffusion_model.load_model({**self.model_args, 'ckpt_path': path, 'pretrain': True}, self.device)


def test_bridge_lstm_tensors():
    """Simple test to verify the BridgeLSTM controller tensor passing"""
    print("Testing BridgeLSTM controller tensor passing...")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    batch_size = 2
    state_dim = 10
    force_dim = 3
    horizon = 16
    hidden_dim = 256
    lstm_hidden_dim = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create mock model_args
    model_args = {
        'action_dim': state_dim,
        'horizon': horizon,
        'interpolant_type': 'linear',  # Interpolation type for bridge diffusion
        'gamma_type': '2^0.5*t(t-1)',  # Noise schedule for diffusion
        'epsilon_type': '1-t',  # Drift schedule for diffusion
        'prior_policy': 'vla',  # Use VLA as prior (source actions)
        'beta_max': 0.03,  # Maximum noise scale
        'sde_type': 'vs',  # Use velocity-score SDE
        'obs_dim': 256,  # Dimension of observations
        'obs_horizon': 1,  # Single observation frame
        'net_type': 'unet1D_si',  # Network type
        'pretrain': False,  # Not using pretrained weights
        'context_frames': 2,  # Number of context frames
    }

    # Create controller
    controller = BridgeLSTMController(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        device=device,
        model_args=model_args,
        force_dim=force_dim
    )

    # Mock the stats
    controller.stats = {
        'vla_mins': torch.zeros(state_dim, device=device),
        'vla_maxs': torch.ones(state_dim, device=device),
        'vla_range': torch.ones(state_dim, device=device),
        'action_mins': torch.zeros(state_dim, device=device),
        'action_maxs': torch.ones(state_dim, device=device),
        'action_range': torch.ones(state_dim, device=device)
    }

    # Create random tensors
    current_state = torch.randn(batch_size, state_dim, device=device)
    vla_actions = torch.randn(batch_size, horizon, state_dim, device=device)
    forces = torch.randn(batch_size, horizon, force_dim, device=device)

    # Create LSTM hidden state
    h0 = torch.zeros(2, batch_size, lstm_hidden_dim, device=device)
    c0 = torch.zeros(2, batch_size, lstm_hidden_dim, device=device)
    lstm_hidden = (h0, c0)

    # Mock diffusion model's sample method
    controller.diffusion_model.sample = lambda x_prior, cond, diffuse_step: x_prior

    # Mock encode_images and encode_observation
    controller.encode_images = lambda img1, img2: (
        torch.zeros(batch_size, controller.latent_obs_dim, device=device),
        torch.zeros(batch_size, controller.latent_obs_dim, device=device)
    )
    original_encode_observation = controller.encode_observation
    controller.encode_observation = lambda state, img1, img2, forces: torch.zeros(batch_size, hidden_dim, device=device)

    # Test predict_step
    print("\nTesting predict_step...")
    diffusion_action = torch.randn(batch_size, state_dim, device=device)
    force = torch.randn(batch_size, force_dim, device=device)


    tac_delta_action, next_lstm_hidden = controller.predict_step(
        diffusion_action=diffusion_action,
        force=force,
        lstm_hidden=lstm_hidden
    )
    print(f"✓ predict_step successful")
    print(f"  - Input shapes: diffusion_action {diffusion_action.shape}, force {force.shape}")
    print(f"  - Output shapes: tac_delta_action {tac_delta_action.shape}, hidden {next_lstm_hidden[0].shape}")


    # Test predict_sequence
    print("\nTesting predict_sequence...")
    refined_actions = controller.predict_sequence(
        state=current_state,
        vla_actions=vla_actions,
        force=forces
    )
    print(f"✓ predict_sequence successful")
    print(f"  - Input shapes: state {current_state.shape}, vla_actions {vla_actions.shape}, forces {forces.shape}")
    print(f"  - Output shape: refined_actions {refined_actions.shape}")



if __name__ == "__main__":

    test_bridge_lstm_tensors()