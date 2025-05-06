# GPT Reward
See Project Proposal.pdf for details

# Getting set up
Run pip3 install -r requirements.txt

# documentation
Gymnasium - https://gymnasium.farama.org/environments/mujoco/
PyTorch - https://pytorch.org/get-started/locally/
Robot models - https://github.com/google-deepmind/mujoco_menagerie/tree/main/boston_dynamics_spot

# update requirements file for new packages added
run 
```
pip3 freeze >> requirements.txt
```

# running spot in interactive mode
 python -m mujoco.viewer --mjcf=./robots/boston_dynamics_spot/scene_arm.xml