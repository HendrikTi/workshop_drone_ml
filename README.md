# Installation


Execute all following commands from the windows cmd: 
Depending on how python is installed, one might use
**"py" or "python3" instead of "python"** in the following commands.
You can check your python version using:

```bash
python --version
```

## Create a virtual environment
```bash
python -m venv virt_env/
```

## Enter virtual environment
```bash
.\virt_env\Scripts\activate
``` 


## Install required packages
```bash
pip install -r requirements.txt
```

## Test if all required modules are available
following command must return "SUCCESS" if all went fine.
```bash
python scripts\test.py
```
