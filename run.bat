@echo off
start "Flask Server" cmd /k "flask run --host=0.0.0.0 --port=5000"
start "Ngrok Tunnel" cmd /k "ngrok http 5000 --hostname=intensely-heroic-ape.ngrok-free.app"

echo Press any key to terminate Flask and Ngrok...
pause

taskkill /F /IM "python.exe" /T 
taskkill /IM "ngrok.exe" /T /F

rem Close the Flask terminal window
taskkill /FI "WINDOWTITLE eq Flask Server" /T /F

rem Close the Ngrok terminal window
taskkill /FI "WINDOWTITLE eq Ngrok Tunnel" /T /F
