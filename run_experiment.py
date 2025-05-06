import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import mujoco
import os

# Ensure videos directory exists
os.makedirs("videos", exist_ok=True)

class LocoManipulationEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("./robots/boston_dynamics_spot/scene_arm.xml")
        self.data = mujoco.MjData(self.model)

        # Setup render context for offscreen rendering
        if render_mode == "rgb_array":
            self.frame_width = 640
            self.frame_height = 480
            self._renderer = mujoco.Renderer(self.model, width=self.frame_width, height=self.frame_height)

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.nq + self.nv,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0],
            high=self.model.actuator_ctrlrange[:, 1],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        reward = -np.linalg.norm(obs)
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        elif self.render_mode == "human":
            if not hasattr(self, "viewer"):
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # Human rendering happens in the background viewer
            return None

    def close(self):
        if hasattr(self, "_renderer"):
            self._renderer.close()
        if hasattr(self, "viewer"):
            self.viewer.close()

# Register the environment
gym.register(
    id="LocoManipulation-v0",
    entry_point=LocoManipulationEnv,
    max_episode_steps=1000,
)

# Create environment with video recording
env = gym.make("LocoManipulation-v0", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="./videos",
    episode_trigger=lambda episode_id: True,
    name_prefix="loco_manipulation"
)
env = RecordEpisodeStatistics(env)

# Run sample episodes
for ep in range(4):
    print(f"Starting episode {ep + 1}")
    obs, _ = env.reset()
    done = False
    reward_total = 0
    step_count = 0

    while not done and step_count < 100:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        reward_total += reward
        done = terminated or truncated
        step_count += 1

    print(f"Episode {ep + 1} ended with reward {reward_total:.2f}")

env.close()
print("Finished. Videos saved in ./videos.")