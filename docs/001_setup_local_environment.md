# 1 Setup local environment

## 1.1 Pyenv Installation
Install `pyenv` to manage multiple Python versions. This allows easy switching between versions
and ensures compatibility with Google Colab's default version.
See the [pyenv documentation](https://github.com/pyenv/pyenv) for more details.

```shell
brew update
brew install pyenv

echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

source ~/.zshrc

```

## 1.2 Venv setup
Set up a virtual environment for local development.

```shell
pyenv install 3.10.12  # Default version used on Google Colab for compatibility

python -m venv .venv
source ./.venv/bin/activate

pip install --upgrade pip

# Can't use pip-sync, because pip-tools is most likely not installed yet
pip install -r requirements.txt
```

## 1.3 Refreshing `requirements.txt` after adding dependencies
Use pip-compile to manage dependencies and pin versions to ensure consistency across environments.
Refer to the [pip-tools documentation](https://pypi.org/project/pip-tools/) for more details.

For dependency pinning we use: https://pypi.org/project/pip-tools/
```shell
pip-compile requirements.in
```
