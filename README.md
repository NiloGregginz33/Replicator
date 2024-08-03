# Replicator
This replicator was inspired by Star Trek and a desire to literally "speak" something into existence. The procedure on how to use this app is pretty straight forward. Note: Do Not Attempt with Filaments other than PLA without changing the appropriate temp settings.
1. Root your 3D printer and install moonraker (fluidd is also recommended)
2. Git clone this repository
3. Install ngrok, go to dashboard to find api key
4. pip install -r requirements.txt
5. python setup.py (use tje api from step 3 as your input for the 2nd prompt)
6. python main.py
7. Set up config for ngrok (I believe it's ngrok start --config=config.yml example)
8. flask run --host=0.0.0.0 --port=5000
9. ngrok http -hostname=example.ngrok-free.app 5000
10. Pat yourself on the back! It should work, if not, leave the bug and I will try to fix
