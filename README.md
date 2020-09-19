## STORM Toehold App

To run locally, download the folder and follow instructions below. This virtual environment and packages have only been tested on a Mac running Mojave, so no guarantees if you have another system or OS.

1. Make a virtual environment with conda and python 3.6. Navigate to inside the app folder before pip installing.
```
    conda create -n stormapp python=3.6 anaconda
    conda activate stormapp
    pip3 install -r requirements.txt
```

2. Use flask to run the app. The app will be running on http://127.0.0.1:5000/.
```
    python3 -m flask run
```

3. If the app is failing to move your job from "queued" to "started", it might help to debug your redis queue (within the virtual environment).
```
    rqworker --url redis://localhost:6379 default
```

4. When done with the app, close the localhost window and deactivate the virtual environment.
```
    conda deactivate
```