# Replicator
proof of concept for a star trek replicator, other tests need to be performed. A replicator in the show could make anything from a simple verbal command to quite literally speak something into existence.
In order to use this, a printer capable of running moonraker and capable of using wifi/lan. The first step to use this is to root your 3D printer and install moonraker. Several guides exist, and I wont discuss them here as it depends on the model of printer you have. Second step is to set up an ngrok account. Thirdly, you'll need to install shap-e into this directory as well as install the requirements.txt files. Next you'll want to configure your code so that all the relavent details are there such as your 3d printer ip and config settings. After this, make sure to remove "github" from the file names, I simply renamed them so I dont get confused. Finally, to get it voice activated simply use a ios shortcut to curl the generate_gcode_flask function and open url for the status endpoint. Make sure to include the appropirate headers and for the open url just pass the api key as an arg in the url.


BUGS THAT I EXPERIENCED THAT MIGHT HELP (W HARDWARE):
filament kept breaking in extruder - make sure config is ok
printer cant find wifi - wait it out and see if it shows up on LAN anyways

