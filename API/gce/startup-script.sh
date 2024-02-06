set -v

# Install logging monitor. The monitor will automatically pickup logs sent to
# syslog.
# [START logging]
curl -s "https://storage.googleapis.com/signals-agents/logging/google-fluentd-install.sh" | bash
service google-fluentd restart &
# [END logging]

# Install dependencies from apt
apt-get update
apt-get install -yq \
    git build-essential supervisor python python-dev python-pip libffi-dev libssl-dev virtualenv

#Install GDAL dependency
export DEBIAN_FRONTEND=noninteractive
sudo apt-get -yq install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# pip from apt is out of date, so make it update itself and install virtualenv.
#pip install --upgrade pip virtualenv

# Get the source code from git
# git requires $HOME and it's not set during the startup script.
export HOME=/root
git clone https://ciarankenny1:81Nedloh@github.com/LandPotential/LandPKS_API_SoilID.git /opt/app/soilid

#download the data file
gsutil -m cp -r gs://soilid-global/data /opt/app/soilid
chown -R pythonapp:pythonapp /opt/app/soilid/data

# Create a pythonapp user. The application will run as this user.
useradd -m -d /home/pythonapp pythonapp

chown -R pythonapp:pythonapp /opt/app

#Create a virtualenv for everything to run in
virtualenv /opt/app/soilid/env

/opt/app/soilid/env/bin/pip install -r /opt/app/soilid/requirements.txt
/opt/app/soilid/env/bin/pip install gdal==1.10.0 --global-option=build_ext --global-option="-I/usr/include/gdal/"

export FLASK_CONFIG=production

# Configure supervisor
cat >/etc/supervisor/conf.d/python-app.conf << EOF
[program:pythonapp]
directory=/opt/app/soilid
command=/opt/app/soilid/env/bin/python /opt/app/soilid/app.py
autostart=true
autorestart=true
user=pythonapp
environment=VIRTUAL_ENV="/opt/app/soilid/env",PATH="/opt/app/soilid/env/bin",HOME="/home/pythonapp",USER="pythonapp",FLASK_CONFIG="production"
EOF

supervisorctl reread
supervisorctl update
