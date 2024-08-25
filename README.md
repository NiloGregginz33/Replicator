# Replicator
This replicator was inspired by Star Trek and a desire to literally "speak" something into existence. The procedure on how to use this app is pretty straight forward. Note: Do Not Attempt with Filaments other than PLA without changing the appropriate temp config. Please install python 3.9/3.10 and git beforehand, and consider using a virtual environment if this is not your default python version.
1. Root your 3D printer and install moonraker (fluidd is also recommended)
2. Git clone this repository
3. Install ngrok, go to dashboard to find api key
4. PIP COMMANDS (run seperately and in order): pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --upgrade --force-reinstall | pip install wheel | pip install -r requirements.txt (idk why torch doesnt download otherwise)
5. python setup.py (use the api from step 3 as your input for the 2nd prompt)(DO NOT WORRY if program doesnt execute all the way, it has still done what it needs to)
6. python main.py
7. Set up config for ngrok (I believe it's ngrok start --config=config.yml example | you might need to open multiple terminals in this directory)
8. Set up config for slicer for whatever material by exporting/overwriting your slicer config into config.ini
9. flask run --host=0.0.0.0 --port=5000
10. ngrok http -hostname=example.ngrok-free.app 5000
11. Shortcuts: use the Get_contents_of_url/Open url or android macro equal. I will only provide the ios version here.
![image_shortcut_1](https://github.com/user-attachments/assets/c32d0c29-251c-4c04-95f1-1543514d6ea6)
![image_shortcut_2](https://github.com/user-attachments/assets/196af37f-f867-4e2e-a896-56ec7d034796)
![image_shortcut_3](https://github.com/user-attachments/assets/7a0cd613-c1ec-403c-9e41-8bd48687fe28)

