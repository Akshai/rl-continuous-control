# Continuous control using Reinforcement Learning
This project is aimed to train a double-jointed arm can move to target locations.

We use [DDPG](https://arxiv.org/pdf/1509.02971) to train the agent.

# Environment details

In this environment, double-jointed arms are trained to move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agents is to maintain its position at the target location for as many time steps as possible. We are training with 20 agents, shared experience replay helps in making the training faster.

#### *State and action space*:

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


# Environment Setup

We utilize [Unity Machine Learning Agents](https://github.com/Unity-Technologies/ml-agents) plugin to interact with the environment. 

To set up your python environment to run the code in this repository, follow the instructions below:

1. Create (and activate) a new [Conda](https://docs.anaconda.com/anaconda/install/) environment with Python 3.6.

    ```bash
    conda create --name <env_name> python=3.6
    conda activate <env_name>
    ```


2. Clone the repository, and navigate to the **python/** folder. Then, install several dependencies.

    ```bash
    git clone https://github.com/Akshai/rl-continuous-control.git
    cd rl-continuous-control/python
    pip install .
    ```
    
    
3.  Download the environment from one of the links below. You need to only select the environment that matches your operating system:

    Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) <br />
    Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)<br />
    Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)<br />
    Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)<br />
    Place the file in the **rl-continuous-control/** folder, and unzip (or decompress) the file.

# Training

From the home path **rl-continuous-control/**, run the following command to train the agent:

```bash
python train.py
```

# Visualization

From the home path **rl-continuous-control/**, run the following command to visualize the trained agent:

```bash
python visualize.py <Path to your trained checkpoint>
```

# Video Demo: <br />
[![Video Demo](https://www.youtube.com/watch?v=l42uVQITrzY/0.jpg)](https://www.youtube.com/watch?v=l42uVQITrzY)





