# Automated Web Application Testing using Deep Reinforcement Learning

MEng thesis, Imperial College London, 2023.

- A Soft Actor-Critic agent learns to test a web application from screenshots alone.
- It runs in a custom Gymnasium environment around the Cypress Real World App.
- Result: 91% better than a Q-learning baseline and 67% better than novice human testers.

  
# Installation Instructions


git clone git@github.com:timeo-schmidt/gui-testing.git
git clone git@github.com:cypress-io/cypress-realworld-app.git

# Setup Cypress Real World App
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash - &&\
sudo apt-get install -y nodejs

npm install yarn -g
cd cypress-realworld-app/
yarn

pip install git+https://github.com/DLR-RM/stable-baselines3

# python deps manually

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install chromedriver (download and unzip

sudo apt update 
sudo apt install -y unzip xvfb libxi6 libgconf-2-4 )
