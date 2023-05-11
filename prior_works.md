Eskonen et al Method:

Visual DRL:

Method:
-> Screenshot of GUI
-> Shrink image to 128x128 using bicubic interpolation
-> Multiple Convolutional layers
-> Flattening to vector
-> Passing to LSTM
-> Forward to A3C (Actor and Critic both 1 FC layer)
-> Combining the "heatmap" of actor and grouping it by exact element coordinates. (Get per element average)

Only does click locations.
Forms are detected using HTML information and random strings are inputted

Reward function is positive when "agent finds something new in the GUI'

Proposed Future Work:

1) More DRL architectures:
- PPO
- Rainbow
- World Models

2) Study the inclusion of more complicated actions (scrolling? drag and drop?) + Choosing what to type in text fields (LLMs?)


Automated Web App Testing using Deep Reinforcement Learning


Minimum:
1 - Debug current appraoch
2 - Implement paper benchmark


Wednesday 1PM


