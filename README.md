# Solving Lunar Lander using Deep Q-Network (DQN)

![Lunar Lander](../../../_images/lunar_lander.gif)

## Lunar Lander Problem Description

The Lunar Lander environment is a classic rocket trajectory optimization problem included in the Box2D environments. The goal is to land a rocket safely on the lunar surface, considering various factors such as fuel consumption, velocity, and orientation. This problem is often addressed using reinforcement learning techniques, and in this project, we focus on solving it using Deep Q-Networks (DQN).

## Environment Overview

### Action Space
- Discrete(4): The agent can take four discrete actions.
   1. Do nothing
   2. Fire left orientation engine
   3. Fire main engine
   4. Fire right orientation engine

### Observation Space
- Box([-1.5, -1.5, -5., -5., -3.1415927, -5., -0., -0.], [1.5, 1.5, 5., 5., 3.1415927, 5., 1., 1.], (8,), float32): The state is an 8-dimensional vector representing the lander's position, velocity, angle, angular velocity, and leg contact status.

### Environment Initialization
```python
import gym
gym.make("LunarLander-v2")
```
## Lunar Lander Dynamics
This environment follows Pontryagin’s maximum principle, making it optimal to fire the engine at full throttle or turn it off. The landing pad is always at coordinates (0, 0), and fuel is infinite, allowing the agent to learn to fly and land on its first attempt.

## Actions and Rewards
### Actions
The agent can take discrete actions to control the orientation and movement of the lander.
### Rewards
The reward is influenced by factors such as proximity to the landing pad, speed, tilt, leg contact, and engine firing.
### Episode Termination
An episode ends if the lander crashes, goes outside the viewport, or is not awake.



## Running the Code

1. Install necessary dependencies.
    ```bash
    # Example installation using pip
    pip install gym
    # Install the requirements
    pip install -r requirements.txt
    ```

2. Run the main training script.
    ```markdown
    python main.py
    ```

3. If the code encounters issues in your coding environment setup, you can run the lunar_lander.ipynb notebook on Google Colab for a seamless execution.
## Result After 2000 episode




