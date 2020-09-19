## STORM Toehold App

To run locally, download the folder and follow instructions below. This virtual environment and packages have only been tested on a Mac running Mojave, so no guarantees if you have another system or OS.

<br>

### 1. Create virtual environment

##### Option #1a (conda)

Make a virtual environment with conda and python 3.6. Navigate to inside the app folder before pip installing.

```bash
conda create -n stormapp python=3.6 anaconda
conda activate stormapp
pip3 install -r requirements.txt
```

##### Option #1b (virtualenv)

Make a Python 3 virtual environment with virtualenv and install requirements.

```bash
python3 -m venv stormapp
source stormapp/bin/activate
pip3 install -r requirements.txt
``` 

<br>

### 2. Run the app locally 

In one tab, start a redis server on the default port 6379.

```bash
redis-server
```

<p>
<img src="README_redis.png" width="400">
</p>

In a second tab (also in the virtual environment), use flask to run the app.

```bash
python3 -m flask run
```

You can now view the app running on http://127.0.0.1:5000/.

When done with the app, close the localhost window and deactivate the virtual environment with `conda deactivate` (if using conda) or `deactivate` (if using virtualenv).

<br>

### Troubleshooting

If the app is failing to move your job from "queued" to "started", it might help to debug your redis queue (within the virtual environment).

```bash
rqworker --url redis://localhost:6379 default
```