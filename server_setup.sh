# This should ideally be a Dockerfile

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