from flask import Flask, request, send_file
from flask_cors import CORS
import Foot_Pressure
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import json
import traceback
import os

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Print current working directory and check if image exists
print("Current working directory:", os.getcwd())
print("Image path:", Foot_Pressure.IMAGE_PATH)
print("Image exists:", os.path.exists(Foot_Pressure.IMAGE_PATH))

# Configure matplotlib for non-interactive use
plt.ioff()

def generate_pressure_map(sensor_data):
   
    try:
        # Clean up any existing plots
        plt.close('all')
        
        # Set the sensor data in Foot_Pressure module
        Foot_Pressure.SENSORS = sensor_data
        
        # Generate the visualization using existing functions
        image, foot_mask, height, width = Foot_Pressure.load_and_process_image(Foot_Pressure.IMAGE_PATH)
        cols, rows = Foot_Pressure.calculate_grid_dimensions(width, height, Foot_Pressure.CELL_SIZE)
        sensor_positions, pressure_values = Foot_Pressure.convert_sensor_coordinates(
            sensor_data, width, height, Foot_Pressure.CELL_SIZE
        )
        pressure_grid = Foot_Pressure.create_pressure_grid(
            foot_mask, sensor_positions, pressure_values,
            rows, cols, Foot_Pressure.CELL_SIZE, width, height
        )
        
        # Create visualization in a new figure
        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(111)
        
        # Setup plot
        pressure_cmap = Foot_Pressure.create_colormap()
        im = Foot_Pressure.setup_plot_appearance(ax, rows, cols, pressure_grid, pressure_cmap)
        Foot_Pressure.draw_grid_lines(ax, rows, cols)
        Foot_Pressure.add_sensor_annotations(ax, sensor_data)
        
        # Add colorbar using the correct figure reference
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Pressure (kg)', rotation=270, labelpad=15)
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf
        
    except Exception as e:
        print(f"Error generating pressure map: {str(e)}")
        raise

@app.route('/generate_map', methods=['POST', 'OPTIONS'])
def generate_map():
    """API endpoint to generate pressure map from sensor data"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        print("Request received:", request.method)
        print("Request headers:", dict(request.headers))
        
        sensor_data = request.json
        if not sensor_data:
            print("No sensor data provided")
            return {'error': 'No sensor data provided'}, 400
            
        print("Received sensor data:", sensor_data)
            
        # Generate pressure map
        try:
            img_buf = generate_pressure_map(sensor_data)
            print("Pressure map generated successfully")  # Debug log
            
            # Convert to base64 for sending to frontend
            img_str = base64.b64encode(img_buf.getvalue()).decode()
            return {'image': f'data:image/png;base64,{img_str}'}
        except Exception as e:
            print("Error in generate_pressure_map:", str(e))  # Debug log
            return {'error': f'Error generating pressure map: {str(e)}'}, 500
        
    except Exception as e:
        print("Error in generate_map endpoint:", str(e))  # Debug log
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(port=5000)
# python DOan/pressure_api.py