from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
import argparse

def process(args):
    NUM_AGENTS = 20
    ACTOR_CHECKPOINT_PATH = args.actor_checkpoint
    CRITIC_CHECKPOINT_PATH = args.critic_checkpoint
    env = UnityEnvironment(file_name="Reacher_multi.app")
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(NUM_AGENTS)                          # initialize the score (for each agent)
    brain = env.brains[brain_name]
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size

    agent = Agent(state_size, action_size,1, actor_checkpoint_path=ACTOR_CHECKPOINT_PATH, critic_checkpoint_path=CRITIC_CHECKPOINT_PATH)

    cnt = 0

    while True:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states  # roll over states to next time step
        cnt+=1
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    env.close()

def get_args():
    parser = argparse.ArgumentParser(description='Visualize the navigation of a trained model')
    parser.add_argument('--actor_checkpoint', help ='Path to the trained actor model', default= 'trained_checkpoints/actor_checkpoint_92.pth')
    parser.add_argument('--critic_checkpoint', help ='Path to the trained critic model',default= 'trained_checkpoints/critic_checkpoint_92.pth')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    process(args)