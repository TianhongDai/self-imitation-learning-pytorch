from arguments import achieve_arguments
from models import Net
import torch
from utils import select_actions
from baselines.common.cmd_util import make_atari_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
import cv2
import numpy as np

# update the current observation
def get_tensors(obs):
    input_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)
    return input_tensor

if __name__ == "__main__":
    args = achieve_arguments()
    # create environment
    env = VecFrameStack(make_atari_env(args.env_name, 1, args.seed), 4)
    # get the model path
    model_path = args.save_dir + args.env_name + '/model.pt'
    network = Net(env.action_space.n)
    network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)) 
    obs = env.reset()
    while True:
        env.render()
        # get the obs
        with torch.no_grad():
            input_tensor = get_tensors(obs)
            _, pi = network(input_tensor)
        actions = select_actions(pi, True)
        obs, reward, done, _ = env.step([actions])
    env.close()
