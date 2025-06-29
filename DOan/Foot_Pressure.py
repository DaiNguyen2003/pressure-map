import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist
from matplotlib.colors import LinearSegmentedColormap

# Configuration constants
CELL_SIZE = 20
POWER = 2
IMAGE_PATH = "DOan/Image/banchan.png"

# Sensor data
SENSORS = {
    'M01': {'position': (22.5, 97), 'pressure': 4.13}, 
    'M02': {'position': (17, 70), 'pressure': 0.512    },
    'M03': {'position': (32, 70), 'pressure': 1.19},
    'M04': {'position': (10, 36 ), 'pressure': 1.04},
    'M05': {'position': (24, 41), 'pressure': 0},
    'M06': {'position': (36, 40), 'pressure': 00.18},
    'M07': {'position': (12, 18), 'pressure': 2.1},
    'M08': {'position': (31, 22), 'pressure': 0.07}
}

def load_and_process_image(image_path):
    """
    Tải và xử lý ảnh để tạo mask của bàn chân
    
    Args:
        image_path (str): Đường dẫn đến file ảnh
        
    Returns:
        tuple: (image, foot_mask, height, width)
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Không thể tải ảnh từ {image_path}")
    
    # Apply binary thresholding
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create foot mask
    foot_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(foot_mask, contours, 255)
    
    height, width = image.shape[:2]
    
    return image, foot_mask, height, width

def calculate_grid_dimensions(width, height, cell_size):
    """
    Tính toán kích thước grid dựa trên kích thước ảnh và kích thước ô
    
    Args:
        width (int): Chiều rộng ảnh
        height (int): Chiều cao ảnh
        cell_size (int): Kích thước ô grid
        
    Returns:
        tuple: (cols, rows)
    """
    cols = width // cell_size
    rows = height // cell_size
    return cols, rows

def convert_sensor_coordinates(sensors_data, image_width, image_height, cell_size):
    """
    Chuyển đổi tọa độ cảm biến từ grid coordinate sang normalized coordinate
    
    Args:
        sensors_data (dict): Dữ liệu cảm biến
        image_width (int): Chiều rộng ảnh
        image_height (int): Chiều cao ảnh
        cell_size (int): Kích thước ô grid
        
    Returns:
        tuple: (sensor_positions, pressure_values)
    """
    sensor_positions = {}
    pressure_values = {}
    
    for sensor_name, data in sensors_data.items():
        gx, gy = data['position']
        pressure = data['pressure']
        
        # Convert grid coord to pixel coord
        pixel_x = gx * cell_size + cell_size // 2
        pixel_y = gy * cell_size + cell_size // 2
        
        # Normalize coordinates to 0-100
        norm_x = (pixel_x / image_width) * 100
        norm_y = ((image_height - pixel_y) / image_height) * 100  # Flip Y axis
        
        sensor_positions[sensor_name] = [norm_x, norm_y]
        pressure_values[sensor_name] = pressure
    
    return sensor_positions, pressure_values

def idw_interpolation(target_points, sensor_positions, pressure_values, power=2):
    """
    Thực hiện nội suy IDW (Inverse Distance Weighting)
    
    Args:
        target_points (np.array): Các điểm cần tính áp suất
        sensor_positions (dict): Vị trí các cảm biến
        pressure_values (dict): Giá trị áp suất của các cảm biến
        power (int): Lũy thừa cho IDW
        
    Returns:
        np.array: Giá trị áp suất được nội suy
    """
    sensor_coords = np.array(list(sensor_positions.values()))
    sensor_values = np.array(list(pressure_values.values()))
    
    distances = cdist(target_points, sensor_coords)
    distances = np.where(distances == 0, 1e-10, distances)
    
    weights = 1.0 / (distances ** power)
    weights_normalized = weights / weights.sum(axis=1, keepdims=True)
    
    interpolated_values = np.sum(weights_normalized * sensor_values, axis=1)
    return interpolated_values

def create_pressure_grid(foot_mask, sensor_positions, pressure_values, 
                        rows, cols, cell_size, width, height):
    """
    Tạo ma trận áp suất cho toàn bộ grid
    
    Args:
        foot_mask (np.array): Mask của bàn chân
        sensor_positions (dict): Vị trí các cảm biến
        pressure_values (dict): Giá trị áp suất của các cảm biến
        rows (int): Số hàng grid
        cols (int): Số cột grid
        cell_size (int): Kích thước ô grid
        width (int): Chiều rộng ảnh
        height (int): Chiều cao ảnh
        
    Returns:
        np.array: Ma trận áp suất
    """
    pressure_grid = np.zeros((rows, cols))
    
    for row in range(rows):
        for col in range(cols):
            # Calculate center coordinates of the cell
            center_x = col * cell_size + cell_size // 2
            center_y = row * cell_size + cell_size // 2
            
            # Check if cell is within image bounds and foot area
            if center_y < height and center_x < width:
                if foot_mask[center_y, center_x] > 0:
                    # Normalize coordinates
                    norm_x = (center_x / width) * 100
                    norm_y = ((height - center_y) / height) * 100
                    
                    # Calculate pressure at this point
                    target_point = np.array([[norm_x, norm_y]])
                    pressure_value = idw_interpolation(
                        target_point, sensor_positions, pressure_values, POWER
                    )[0]
                    pressure_grid[row, col] = pressure_value
                else:
                    pressure_grid[row, col] = np.nan
            else:
                pressure_grid[row, col] = np.nan
    
    return pressure_grid

def create_colormap():
    """
    Tạo colormap cho hiển thị áp suất
    
    Returns:
        LinearSegmentedColormap: Colormap tùy chỉnh
    """
    colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    return LinearSegmentedColormap.from_list('pressure', colors, N=100)

def draw_grid_lines(ax, rows, cols):
    """
    Vẽ các đường grid
    
    Args:
        ax: Matplotlib axis object
        rows (int): Số hàng
        cols (int): Số cột
    """
    for row in range(rows):
        for col in range(cols):
            rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=0.5, #------------------------
                        edgecolor='black', facecolor='none')
            ax.add_patch(rect)

def add_sensor_annotations(ax, sensors_data):
    """
    Thêm annotation cho các cảm biến
    
    Args:
        ax: Matplotlib axis object
        sensors_data (dict): Dữ liệu cảm biến
    """
    for sensor_name, data in sensors_data.items():
        gx, gy = data['position']
        pressure = data['pressure']
        
        # Add sensor name
        ax.text(gx, gy-1.5, sensor_name, ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.8))

        # Add pressure value
        ax.text(gx, gy+1, f'{pressure:.2f}', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='black',
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))

def setup_plot_appearance(ax, rows, cols, pressure_grid, pressure_cmap):
    """
    Thiết lập giao diện plot
    
    Args:
        ax: Matplotlib axis object
        rows (int): Số hàng
        cols (int): Số cột
        pressure_grid (np.array): Ma trận áp suất
        pressure_cmap: Colormap
        
    Returns:
        matplotlib image object: Để tạo colorbar
    """
    ax.set_title("Pressure Map", fontsize=16, fontweight='bold')
    
    # Prepare display grid
    display_grid = np.copy(pressure_grid)
    display_grid = np.where(np.isnan(display_grid), 0, display_grid)
    
    # Display grid with colors
    im = ax.imshow(display_grid, cmap=pressure_cmap, vmin=0, vmax=4.5, aspect='equal')
    
    # Set axis properties
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(rows-0.5, -0.5)
    ax.set_xticks(range(0, cols, 5))
    ax.set_yticks(range(0, rows, 5))
    ax.grid(True, alpha=0.3)
    
    return im

def print_system_info(cols, rows, cell_size, sensors_data):
    """
    In thông tin hệ thống
    
    Args:
        cols (int): Số cột
        rows (int): Số hàng
        cell_size (int): Kích thước ô
        sensors_data (dict): Dữ liệu cảm biến
    """
    print(f"Grid created: {cols} columns x {rows} rows")
    print(f"Each cell: {cell_size}x{cell_size} pixels")
    print("Sensor information:")
    for sensor_name, data in sensors_data.items():
        position = data['position']
        pressure = data['pressure']
        print(f"  {sensor_name}: {pressure} at grid ({position[0]}, {position[1]})")

def main():
    """
    Hàm chính để chạy toàn bộ chương trình
    """
    try:
        # 1. Load and process image
        image, foot_mask, height, width = load_and_process_image(IMAGE_PATH)
        
        # 2. Calculate grid dimensions
        cols, rows = calculate_grid_dimensions(width, height, CELL_SIZE)
        
        # 3. Convert sensor coordinates
        sensor_positions, pressure_values = convert_sensor_coordinates(
            SENSORS, width, height, CELL_SIZE
        )
        
        # 4. Create pressure grid
        pressure_grid = create_pressure_grid(
            foot_mask, sensor_positions, pressure_values, 
            rows, cols, CELL_SIZE, width, height
        )
        
        # 5. Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        
        # 6. Setup plot appearance
        pressure_cmap = create_colormap()
        im = setup_plot_appearance(ax, rows, cols, pressure_grid, pressure_cmap)
        
        # 7. Draw grid lines
        draw_grid_lines(ax, rows, cols)
        
        # 8. Add sensor annotations
        add_sensor_annotations(ax, SENSORS)
        
        # 9. Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Pressure (kg)', rotation=270, labelpad=15)
        
        # 10. Print system information
        print_system_info(cols, rows, CELL_SIZE, SENSORS)
        
        # 11. Show plot
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Lỗi trong quá trình thực thi: {e}")

if __name__ == "__main__":
    main()