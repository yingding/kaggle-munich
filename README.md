# kaggle-munich
the notebook for Kaggle Munich SHAP 101 talk

# Introduction into SHAP

SHAP trains an additional model based on the observation of custom model behaviour to determine how black box custom model making decision regarding the input features.

## Create a virtual environment

For macosx apple silicon
```shell
/opt/homebrew/bin/python3.10 -m venv ~/VENV/shap3.10
```

## Install packages 
```shell
source ~/VENV/shap3.10/bin/activate
python3 -m pip install -r requirements.txt --no-cache
deactivate
```

## Add a jupyter notebook kernel to VENV
```console
source ~/VENV/shap3.10/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install ipykernel
deactivate
```

We need to reactivate the venv so that the ipython kernel is available after installation.
```
source ~/VENV/shap3.10/bin/activate
# ipython kernel install --user --name=shap3.10
python3 -m ipykernel install --user --name=shap3.10 --display-name "shap3.10"

```
Note: 
* restart the vs code, to select the venv as jupyter notebook kernel 
* name is `shap3.10`, which is the venv name.

Reference:
* https://ipython.readthedocs.io/en/stable/install/kernel_install.html
* https://anbasile.github.io/posts/2017-06-25-jupyter-venv/

## Remove ipykernel
```shell
# jupyter kernelspec uninstall -y <VENV_NAME>
jupyter kernelspec uninstall -y shap3.10
```

## Remove all package from venv
```
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```