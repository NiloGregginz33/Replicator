import requests
from zipfile import ZipFile
import os
import subprocess

def write_keys_to_file(filename='secret_k.txt'):
    secret_key = input("Enter app secret key (This can be anything you want): ")
    api_key = input("Enter Flask API key (This must be obtained from your ngrok dashboard) : ")
    local_key = input("Enter local key: (This can be anything you want): ")
    local_ip = input("Enter the local IP for your 3D printer: ")

    with open(filename, 'w') as f:
        f.write(f"app.secret_key={secret_key}\n")
        f.write(f"api_key={api_key}\n")
        f.write(f"local_key={local_key}\n")
        f.write(f"local_ip={local_ip}\n")

    print(f"Keys have been written to {filename}")

def get_full_path(filename):
    current_directory = os.getcwd()   
    full_path = os.path.join(current_directory, filename)
    return full_path

def clone_repository(repo_url, target_directory):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    try:
        subprocess.run(['git', 'clone', repo_url, target_directory], check=True)
        print(f'Repository cloned into {target_directory}')
    except subprocess.CalledProcessError as e:
        print(f'Error occurred while cloning the repository: {e}')

def install_package(target_directory):

    path = get_full_path("venv")
    try:
        os.chdir(target_directory)
        print(f"Changed directory to {os.getcwd()}")
        
        print("Running 'pip install -e .' before setup.py...")
        subprocess.run(['pip', 'install', '-e', '.'], check=True)
        
        print("Running 'python setup.py'...")
        subprocess.run(['python', 'setup.py', 'install', '--install-dir', path], check=True)
        
        print("Running 'pip install -e .' after setup.py...")
        subprocess.run(['pip', 'install', '-e', '.'], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    finally:
        os.chdir('..')
        print(f"Changed back to the original directory: {os.getcwd()}")
        
write_keys_to_file()

extraction_folder = 'slic3r'

if not os.path.exists(extraction_folder):
    os.makedirs(extraction_folder)
    print(f'Created directory: {extraction_folder}')
    
dropbox_url = 'https://github.com/slic3r/Slic3r/releases/download/1.3.0/Slic3r-1.3.0.64bit.zip'

response = requests.get(dropbox_url)

filename = 'slic3r.zip' 
with open(filename, 'wb') as file:
    file.write(response.content)

print(f'File downloaded as {filename}')

with ZipFile(filename, 'r') as zip:
    zip.printdir()
    print("Extracting...")
    zip.extractall(extraction_folder)
    print('Done!')

repo_url = 'https://github.com/openai/shap-e.git'

shape_folder = 'shap-e'

clone_repository(repo_url, shape_folder)

print("Shap-e successfully cloned")

install_package(shape_folder)

