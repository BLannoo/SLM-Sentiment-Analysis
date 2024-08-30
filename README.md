# Setup local environment

## Pyenv installation
```shell
brew update
brew install pyenv

echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

source ~/.zshrc
```

## Venv setup
```shell
pyenv install 3.10.12  # Default version used on Google Collab

python -m venv .venv
source ./.venv/bin/activate

pip install --upgrade pip

# Can't use pip-sync, because pip-tools is most likely not installed yet
pip install -r requirements.txt
```

## Refreshing requirements.txt after adding dependencies
```shell
# To ensure consistency across environments, 
# generate the requirements file constrained by the packages available in Google Colab.
pip-compile requirements.in -c requirements-collab-freeze.txt
```
