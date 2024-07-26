import os
import sys
import torch
from PIL import Image
import numpy as np
import requests
import time
import json
import subprocess
import socket
import trimesh
from flask import Flask, request, jsonify, abort, Response, render_template
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
from functools import wraps
from pyngrok import ngrok, conf
import logging
import pymeshlab
from scipy.interpolate import splprep, splev
import threading
import logging
import open3d as o3d
import re
import random
import trimesh
import cv2
from scipy.interpolate import splprep, splev
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh


root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Add the shap-e directory to the Python path
shap_e_path = os.path.join(os.path.dirname(__file__), 'shap-e')
sys.path.append(shap_e_path)

ip = "{INSERT YOUR LOCAL PRINTER IP"
MOONRAKER_IP = f'http://{ip}:7125'
HEADERS = {'Content-Type': 'application/json'}
port = 7125

generation_status = {
    "status": "not_started",
    "percentage": 0,
    "images": []
}

app = Flask(__name__)

API_KEY = "{INSERT YOUR NGROK AUTH KEY}"
local_key = "{INSERT ANY SECURE STRING HERE FOR YOUR OWN API TOKEN SO OTHER PEOPLE DONT USE YOUR STUFF"


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    
folder_name = "gcode"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")


def detect_floating_parts(mesh, min_component_size=50):
    components = mesh.split(only_watertight=False)
    significant_parts = [comp for comp in components if len(comp.vertices) >= min_component_size]
    return significant_parts

def create_spline_connection(point1, point2, num_points=100):
    if np.array_equal(point1, point2):
        raise ValueError("The points are identical; cannot create a spline connection.")
    
    mid_point = (point1 + point2) / 2
    mid_point[2] += np.linalg.norm(point1 - point2) / 4

    try:
        tck, u = splprep([np.array([point1[0], mid_point[0], point2[0]]),
                          np.array([point1[1], mid_point[1], point2[1]]),
                          np.array([point1[2], mid_point[2], point2[2]])], s=0)
    except Exception as e:
        raise ValueError(f"Failed to create spline: {e}")

    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new, z_new = splev(u_new, tck, der=0)
    return np.vstack((x_new, y_new, z_new)).T

def create_tube_along_spline(spline_points, radius=0.3):
    tube_segments = []
    for i in range(len(spline_points) - 1):
        segment = trimesh.creation.cylinder(radius=radius, segment=[spline_points[i], spline_points[i + 1]])
        tube_segments.append(segment)

    tube = trimesh.util.concatenate(tube_segments)
    return tube

def create_connection(mesh1, mesh2):
    closest_points1, distances1, _ = trimesh.proximity.closest_point(mesh1, mesh2.vertices)
    closest_points2, distances2, _ = trimesh.proximity.closest_point(mesh2, mesh1.vertices)

    if distances1.size == 0 or distances2.size == 0:
        raise ValueError("No closest points found between meshes")

    point1_index = distances1.argmin()
    point2_index = distances2.argmin()

    if point1_index >= mesh1.vertices.shape[0] or point2_index >= mesh2.vertices.shape[0]:
        raise IndexError("Closest point index out of bounds")

    point1 = mesh1.vertices[point1_index]
    point2 = mesh2.vertices[point2_index]

    spline_points = create_spline_connection(point1, point2)
    tube = create_tube_along_spline(spline_points)

    return tube

def bind_floating_parts(input_filepath, output_filepath):
    mesh = trimesh.load(input_filepath)
    parts = detect_floating_parts(mesh)

    if len(parts) <= 1:
        mesh.export(output_filepath)
        return

    merged_mesh = parts[0]
    for part in parts[1:]:
        try:
            connection = create_connection(merged_mesh, part)
            merged_mesh = trimesh.util.concatenate([merged_mesh, part, connection])
        except Exception as e:
            print(f"Failed to create connection: {e}")

    merged_mesh.export(output_filepath)


def start_ngrok():
    os.environ["NGROK_CONFIG"] = "ngrok.yml"
    public_url = ngrok.connect(5000, "http")
    ngrok_process = ngrok.get_ngrok_process()
    print("Ngrok Tunnel URL:", public_url)
    print(f"Process: {ngrok_process}")
    logger.info(f" * ngrok tunnel \"{public_url}\" ->\" http:127.0.0.1:5000\"")

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key') or request.args.get('api_key')
        if api_key or api_key == local_key:
            return f(*args, **kwargs)
        else:
            abort(401)
    return decorated_function
        
def get_temperature():
    url = f"{MOONRAKER_IP}/printer/objects/query?extruder=temperature&heater_bed=temperature"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        print("Temperature info retrieved successfully.")
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"Failed to retrieve temperature info: {response.content}")

def wait_for_temperature(target_temp, target_bed_temp, timeout=420):
    start_time = time.time()
    while time.time() - start_time < timeout:
        url = f"{MOONRAKER_IP}/printer/objects/query?extruder=temperature&heater_bed=temperature"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            extruder_temp = data.get('result', {}).get('status', {}).get('extruder', {}).get('temperature', 0)
            bed_temp = data.get('result', {}).get('status', {}).get('heater_bed', {}).get('temperature', 0)
            print(f"Extruder: {extruder_temp}°C, Bed: {bed_temp}°C")
            if extruder_temp >= target_temp and bed_temp >= target_bed_temp:
                print("Target temperatures reached.")
                return True
        time.sleep(5)
    print("Timeout reached before temperatures were met.")
    return False
    
# Function to set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def prompt_to_gcode(prompt, output_gcode_path, batch_size=1, guidance_scale=20.0, render_mode='nerf', size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(random.randint(0, 10000000))
    # Load models and configurations
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    slic3r_path = "prusa3d.exe"
    profile_path = "config.ini"

    # Sample latents
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=20,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    image_dir = os.path.join('images', prompt.replace(" ", "_"))
    os.makedirs(image_dir, exist_ok=True)

    # Create cameras and render images
    cameras = create_pan_cameras(size, device)
    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        # Save the first image of each latent as PNG
        for j, image in enumerate(images):
            image_path = os.path.join(image_dir, f'image_{i}_{j}.png')
            image.save(image_path)
            print(f"Saved {image_path}")

    # Save the latents as meshes and convert to G-code
    gcode_dir = os.path.join(os.path.dirname(output_gcode_path), 'gcodes')
    os.makedirs(gcode_dir, exist_ok=True)
    gcode_path = os.path.join(gcode_dir, f'{prompt.replace(" ", "_")}.gcode')
    
    
    for i, latent in enumerate(latents):
        base = f'{prompt.replace(" ", "_")}_repaired.gcode'
        
        t = decode_latent_mesh(xm, latent).tri_mesh()
        obj_path = f'{prompt.replace(" ", "_")}_{i}.obj'
        print("creating mesh...")
        with open(obj_path, 'w') as f:
            t.write_obj(f)
        print(f"Saved {obj_path}")
        
        # Convert OBJ to G-code
        
        obj_path_1 = f'{prompt.replace(" ", "_")}_halfway.obj'
        bind_floating_parts(obj_path, obj_path_1)
        obj_to_gcode(obj_path_1, gcode_path, prompt, slic3r_path, profile_path)
        
        print(f"Saved G-code to {gcode_path}")
    
    return gcode_dir

def scale_obj_file(input_filepath, output_filepath, scale_factor=60.0, target_size_cm=5):
    try:
        start_time = time.time()
        
        # Load the OBJ file
        print("Loading OBJ file...")
        mesh = trimesh.load(input_filepath)
        load_time = time.time()
        print(f"OBJ file loaded in {load_time - start_time:.2f} seconds.")

        extents = mesh.extents
        max_extent = max(extents)

        target_scale = (target_size_cm * 10) / max_extent
        
        # Scale the vertices of the mesh
        print("Scaling the mesh...")
        mesh.apply_scale(target_scale)
        scale_time = time.time()
        print(f"Mesh scaled in {scale_time - load_time:.2f} seconds.")
        
        # Export the scaled mesh to a new OBJ file
        print("Exporting the scaled mesh to OBJ file...")
        mesh.export(output_filepath)
        export_time = time.time()
        print(f"Scaled mesh exported in {export_time - scale_time:.2f} seconds.")
        
        print(f"Successfully scaled {input_filepath} by {scale_factor} and saved to {output_filepath}")
        total_time = time.time()
        print(f"Total execution time: {total_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_gcode(vertices, faces, layer_height=0.2, extrusion_width=0.4, feed_rate=1200):
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    current_z = min_z
    gcode_lines = ["G21 ; Set units to millimeters", "G90 ; Absolute positioning", "G1 E10 F100 ; Prime extruder"]

    while current_z <= max_z:
        layer_vertices = vertices[np.abs(vertices[:, 2] - current_z) < layer_height / 2]
        if len(layer_vertices) == 0:
            current_z += layer_height
            continue

        # Sort vertices to create a simple path
        layer_vertices = layer_vertices[np.argsort(layer_vertices[:, 0])]
        
        gcode_lines.append(f"G1 Z{current_z:.3f} F{feed_rate}")

        for i, vertex in enumerate(layer_vertices):
            x, y, z = vertex
            if i == 0:
                gcode_lines.append(f"G1 X{x:.3f} Y{y:.3f} F{feed_rate}")
            else:
                gcode_lines.append(f"G1 X{x:.3f} Y{y:.3f} E{extrusion_width}")

        current_z += layer_height

    return "\n".join(gcode_lines)

def obj_to_gcode(obj_file_path, output_gcode_path,  prompt, slic3r_path, profile_path, output_filename = "sliced_obj.obj"):
    slic3r_path = r'{INSERT FULL PATH TO SLIC3R}'
    # Update the path based on your installation
    profile_path = r'{INSERT FULL PATH TO CONFIG HERE}'
    ouput = os.path.join("gcodes", f"{prompt.replace(' ', '_')}.gcode")
    filepath = "sliced_obj.obj"
    
    scale_obj_file(obj_file_path, output_filename)
    # Construct the command for Slic3r
    command = [
        slic3r_path,
        '--load', profile_path,  # Load the slicing profile
        '--output', output_gcode_path,
        '--support-material',
        '--no-gui',# Output G-code file
        filepath  # Input OBJ file       
        # Enable support material
    ]

    # Print the command to debug
    print("Running command:", " ".join(command))
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Extract the progress percentage from the output
                print(output.strip())  # Print raw output for debugging
                match = re.search(r'(\d+)%', output)
                if match:
                    percent_done = match.group(1)
                    print(f"Progress: {percent_done}%")
        
        process.wait()
        if process.returncode == 0:
            print(f"G-code successfully generated: {output_gcode_path}")
        else:
            print(f"Error during slicing, exit code: {process.returncode}")
            print("Slicing error output:", process.stderr.read())
    except subprocess.CalledProcessError as e:
        print(f"Error during slicing: {e}")
        print("Slicing error output:", e.stderr)
        
def send_request_with_retries(url, headers=None, json=None, data=None, max_retries=5, delay=5):

    for attempt in range(max_retries):
        try:
            if json:
                response = requests.post(url, headers=headers, json=json)
            elif data:
                response = requests.post(url, headers=headers, data=data)
            else:
                response = requests.get(url, headers=headers)

            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise requests.exceptions.RequestException(f"Failed to send request after {max_retries} attempts")

def send_gcode_batch(printer_ip, port, commands, batch_size=10):
    responses = []
    url = f"http://{ip}:7125/printer/gcode/script"
    headers = {'Content-Type': 'application/json'}
    print(commands)
    if len(commands) == 0:
        print("no commands")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((printer_ip, port))

            for i in range(0, len(commands), batch_size):
                batch = commands[i:i+batch_size]
                script="\n".join(batch)
                json_data = {"script": script}
                try:
                    send_request_with_retries(url, headers=headers, json=json_data, max_retries=1)
                    print("Sent batch: \n {script}")
                except requests.exceptions.RequestException as e:
                    print("Failed batch: {e}")


    except Exception as e:
        responses.append(f"An error occurred: {e}")

    pass

def join_with_current_path(relative_path):
    current_path = os.getcwd()
    combined_path = os.path.join(current_path, relative_path)
    return combined_path

def split_gcode_file(filepath, prompt):
    # List to hold the G-code commands
    gcode_commands = []

    filepath = normalize_path(filepath)
    path = f'gcodes\\{prompt.replace(" ","_")}.gcode'

    # Read the G-code file
    with open(path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace characters
            command = line.strip()
            # Ignore empty lines and comments
            if command and not command.startswith(';'):
                gcode_commands.append(command)

    return gcode_commands

def normalize_path(path):
    # Convert the path to a string (if it's not already)
    path_str = str(path)
    normalized_path = path_str.replace('\\', '/')
    
    return normalized_path


def convert_obj_to_stl(obj_filepath, stl_filepath):
    try:
        # Load the OBJ file
        mesh = trimesh.load(obj_filepath)
        
        # Export the mesh to an STL file
        mesh.export(stl_filepath)
        print(f"Successfully converted {obj_filepath} to {stl_filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

def warm_up():
    base_url = f"http://{ip}"
    try:
        send_request_with_retries(f"{base_url}/printer/gcode/script", json={"script": "M104 S265"})  # Set extruder temperature
        send_request_with_retries(f"{base_url}/printer/gcode/script", json={"script": "M140 S100"})  # Set bed temperature
        print("Extruder and bed temperatures set successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to set temperatures: {e}")

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
@require_api_key
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
@require_api_key
def status():
    return render_template('status.html', api_key=local_key)

@app.before_request
def log_request_info():
    logger.info('Headers: %s', request.headers)
    logger.info('Body: %s', request.get_data())

@app.route("/")
@require_api_key
def homebase():
    return render_template("index.html")
    
@app.route('/generate_gcode', methods=['POST'])
@require_api_key
def generate_gcode_flask():    
    data = request.get_json()
    base_url = f"http://{ip}:7125"
    print(data["prompt"])
    prompt = data["prompt"]
    
    print(f"data: {data}", sys.stderr)
    if not data or 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400

    relative_path = os.path.join('gcodes',f'{prompt.replace(" ", "_")}.gcode')

    output_gcode_path = f'{prompt.replace(" ", "_")}.gcode'

    path = join_with_current_path(relative_path)

    def generate_gcode():
        gcode_path = prompt_to_gcode(prompt, output_gcode_path)
        commands = split_gcode_file(gcode_path, prompt)
        try:
            send_request_with_retries(f"{base_url}/printer/gcode/script", json={"script": "M104 S260"})  # Set extruder temperature
            send_request_with_retries(f"{base_url}/printer/gcode/script", json={"script": "M140 S100"})  # Set bed temperature
            print("Extruder and bed temperatures set successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to set temperatures: {e}")

        # Send G-code commands in batches
        time.sleep(2)
        get_temperature()
        initial_commands = [
            "G21 ; Set units to millimeters",
            "M82 ; Use absolute mode for extrusion",
            "G28 ; Home all axes",
            "M104 S260 ; Set extruder",
            "M140 S100 ; Set bed",
            "M109 S259 ; Wait for extruder to reach target temperature",
            "M190 S99 ; Wait for bed to reach target temperature",
            "G92 E0 ; Reset extruder position",
            "G1 E10 F100 ; Prime the extruder"
        ]
        # Combine initial commands with the actual print commands
        
        full_commands = initial_commands + commands
        print_function(full_commands)

    threading.Thread(target=generate_gcode).start()

    return '', 204


@app.route('/process_text', methods=['POST'])
@require_api_key
def process_text():
    data = request.get_json()
    if not data or 'text' not in data:
        logger.error('No text provided in request')
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    # Log the received text
    logger.info(f"Received text: {text}")
    print(f"{text}", file=sys.stderr)

    # Respond with a confirmation message
    return jsonify({'message': f'Text received: {text}'}), 200

    
def print_function(commands, ip=ip, port=port):
    base_url = f"http://{ip}:7125"
    try:
        send_request_with_retries(f"{base_url}/printer/gcode/script", json={"script": "M104 S230"})  # Set extruder temperature
        send_request_with_retries(f"{base_url}/printer/gcode/script", json={"script": "M140 S100"})  # Set bed temperature
        print("Extruder and bed temperatures set successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to set temperatures: {e}")
    # Send G-code commands in batches
    time.sleep(10)
    get_temperature()
    if wait_for_temperature(229,99):
        thread = threading.Thread(target=send_gcode_batch, args=(ip, port, commands,))
        thread.start()
        thread.join()
    else:
        print("printer failed to heat")
    
def run_flask():
    app.run(port=5000)

if __name__ == "__main__":
    ngrok_thread = threading.Thread(target=start_ngrok)
    ngrok_thread.start()
    run_flask()
    ngrok_thread.join()
    print("Thank you for choosing replicator")
    
