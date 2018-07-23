# tensorflow-test

## Installation
```bash
# Install python3 to /usr/local/bin with pip3
$ brew install python

# Install virtualenv to /usr/local/lib/python3.6
$ pip3 install virtualenv

# Create a virtual env with your installed python version
# Here python3 is the real python exe that brew has installed
$ virtualenv -p python3 .

# Activate the environment
$ . ./bin/activate

# Now we can use pip, which links to the pip inside this folder
# We can install tensorflow inside this folder
$ pip install tensorflow

# Or we install all the libraries we need using 
$ pip install -r requirements.txt

# All the libraries will be installed in ./lib
```