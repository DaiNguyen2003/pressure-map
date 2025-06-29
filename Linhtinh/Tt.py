import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# Configuration constants
CELL_SIZE = 20
POWER = 2
IMAGE_PATH = "feet-outline-paper-crafts.png"
EXCEL_PATH = "Dataa.xlsx"  # Tên file từ artifact

# Vị trí cố định của các cảm biến (grid coordinates)
SENSOR_POSITIONS = {
    'M01': (22, 95), 
    'M02': (14, 75),
    'M03': (32, 75),
    'M04': (8, 40),
    'M05': (22, 40),
    'M06': (38, 38),
    'M07': (11, 14),
    'M08': (31, 15)
}

class PressureMapVideoPlayer:
    def __init__(self, excel_path, image_path):
        self.excel_path = excel_path
        self.image_path = image_path
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = True
        self.animation = None
        
        # Load và setup data
        self.load_excel_data()
        self.setup_image_processing()
        self.create_single_window()
        
    def load_excel_data(self):
        """Đọc dữ liệu từ file Excel được tạo từ artifact"""
        try:
            # Thử đọc file Excel với các engine khác nhau
            engines = ['openpyxl', 'xlrd', None]
            df = None
            
            for engine in engines:
                try:
                    if engine:
                        df = pd.read_excel(self.excel_path, engine=engine)
                    else:
                        df = pd.read_excel(self.excel_path)
                    print(f"✓ Đọc file Excel thành công với engine: {engine or 'default'}")
                    break
                except Exception as e:
                    print(f"⚠ Thử engine {engine}: {e}")
                    continue
            
            if df is None:
                raise Exception("Không thể đọc file Excel với bất kỳ engine nào")
            
            # Hiển thị thông tin file
            print(f"✓ Dữ liệu gốc: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"✓ Các cột: {list(df.columns)}")
            
            # Xử lý các định dạng file khác nhau
            sensor_cols = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08']
            
            # Trích xuất dữ liệu cảm biến
            available_sensors = []
            for sensor in sensor_cols:
                if sensor in df.columns:
                    available_sensors.append(sensor)
                else:
                    # Thử tìm cột tương tự
                    for col in df.columns:
                        if sensor.lower() in col.lower() or col.lower() in sensor.lower():
                            available_sensors.append(col)
                            print(f"✓ Ánh xạ {sensor} -> {col}")
                            break
                    else:
                        available_sensors.append(None)
                        print(f"⚠ Không tìm thấy cột {sensor}")
            
            # Tạo dataframe cảm biến với dữ liệu sạch
            sensor_data = {}
            for i, sensor in enumerate(sensor_cols):
                col_name = available_sensors[i]
                if col_name and col_name in df.columns:
                    # Làm sạch dữ liệu - chuyển về số
                    values = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
                    sensor_data[sensor] = values.values
                else:
                    # Tạo dữ liệu 0 cho cảm biến thiếu
                    sensor_data[sensor] = np.zeros(len(df))
            
            self.sensor_data = pd.DataFrame(sensor_data)
            self.total_frames = len(self.sensor_data)
            
            # Thống kê dữ liệu
            print(f"\n📊 THỐNG KÊ DỮ LIỆU:")
            print(f"   Tổng số frames: {self.total_frames}")
            print(f"   Áp suất max: {self.sensor_data.max().max():.2f}")
            print(f"   Áp suất min: {self.sensor_data.min().min():.2f}")
            
        except Exception as e:
            print(f"❌ Lỗi đọc Excel: {e}")
            print("🔄 Tạo dữ liệu mẫu thay thế...")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Tạo dữ liệu mẫu khi không đọc được file Excel"""
        np.random.seed(42)
        frames = 120  # Tương ứng với 1.2s ở 100Hz
        
        # Tạo dữ liệu theo các giai đoạn như trong artifact
        phases = [
            {"name": "Initial contact", "start": 0, "end": 2},
            {"name": "Loading response", "start": 2, "end": 12},
            {"name": "Mid stance", "start": 12, "end": 31},
            {"name": "Terminal stance", "start": 31, "end": 50},
            {"name": "Pre-swing", "start": 50, "end": 62},
            {"name": "Initial swing", "start": 62, "end": 75},
            {"name": "Mid swing", "start": 75, "end": 100},
            {"name": "Terminal swing", "start": 100, "end": 120}
        ]
        
        # Tạo dữ liệu cảm biến theo logic thực tế
        sensor_data = {}
        for sensor in SENSOR_POSITIONS.keys():
            values = []
            for i in range(frames):
                phase_name = self.get_phase_name(i, phases)
                pressure = self.generate_realistic_pressure(sensor, phase_name, i)
                values.append(pressure)
            sensor_data[sensor] = values
        
        self.sensor_data = pd.DataFrame(sensor_data)
        self.total_frames = frames
        
        print(f"✓ Tạo {frames} frames dữ liệu mẫu thực tế")
        print(f"✓ Mô phỏng 8 giai đoạn bước chân")
    
    def get_phase_name(self, frame_idx, phases):
        """Lấy tên giai đoạn theo frame"""
        for phase in phases:
            if phase["start"] <= frame_idx < phase["end"]:
                return phase["name"]
        return "Terminal swing"
    
    def generate_realistic_pressure(self, sensor, phase_name, sample_idx):
        """Tạo áp suất thực tế theo giai đoạn và cảm biến"""
        base_pressure = 0.0
        noise = np.random.normal(0, 0.05)  # Nhiễu nhỏ
        
        # Logic áp suất theo từng cảm biến và giai đoạn
        if phase_name == "Loading response":
            if sensor == "M01":  # Gót chân
                base_pressure = 2.0 + sample_idx * 0.2
            elif sensor == "M02":
                base_pressure = 0.2
        
        elif phase_name == "Mid stance":
            if sensor == "M01":
                base_pressure = 4.0
            elif sensor == "M02":
                base_pressure = 1.5
            elif sensor == "M03":
                base_pressure = 1.2
            elif sensor == "M04":
                base_pressure = 0.3
        
        elif phase_name == "Terminal stance":
            if sensor == "M01":
                base_pressure = 3.5
            elif sensor == "M02":
                base_pressure = 2.5
            elif sensor == "M03":
                base_pressure = 2.0
            elif sensor == "M04":
                base_pressure = 1.0
            elif sensor == "M05":
                base_pressure = 0.5
        
        elif phase_name == "Pre-swing":
            if sensor in ["M04", "M05", "M06", "M07", "M08"]:  # Mũi chân
                base_pressure = 2.0 + np.random.uniform(0, 1.0)
            elif sensor in ["M01", "M02", "M03"]:  # Gót và giữa chân
                base_pressure = max(0, 1.5 - sample_idx * 0.1)
        
        # Swing phases
        elif "swing" in phase_name.lower():
            base_pressure = 0.0
        
        return max(0, min(4.5, base_pressure + noise))
    
    def setup_image_processing(self):
        """Setup xử lý ảnh và grid"""
        try:
            # Load image
            image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("⚠ Không tìm thấy ảnh, tạo hình bàn chân mặc định")
                image = self.create_default_foot_shape()
            else:
                print(f"✓ Đã load ảnh: {self.image_path}")
            
            # Tạo mask
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            self.foot_mask = np.zeros_like(image, dtype=np.uint8)
            if contours:
                cv2.fillPoly(self.foot_mask, contours, 255)
            else:
                self.foot_mask = np.ones_like(image) * 255
            
            self.height, self.width = image.shape[:2]
            self.cols = self.width // CELL_SIZE
            self.rows = self.height // CELL_SIZE
            
            print(f"✓ Grid: {self.cols}x{self.rows} cells, Image: {self.width}x{self.height}px")
            
        except Exception as e:
            print(f"⚠ Lỗi xử lý ảnh: {e}")
            self.width, self.height = 50 * CELL_SIZE, 120 * CELL_SIZE
            self.foot_mask = self.create_default_mask()
            self.cols, self.rows = 50, 120
    
    def create_default_foot_shape(self):
        """Tạo hình dạng bàn chân mặc định"""
        height, width = 120 * CELL_SIZE, 50 * CELL_SIZE
        image = np.zeros((height, width), dtype=np.uint8)
        
        # Vẽ hình bàn chân đơn giản
        cv2.ellipse(image, (width//2, height//4), (width//3, height//6), 0, 0, 360, 255, -1)  # Ngón chân
        cv2.ellipse(image, (width//2, height*2//3), (width//4, height//3), 0, 0, 360, 255, -1)  # Gót chân
        
        return image
    
    def create_default_mask(self):
        """Tạo mask mặc định"""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.ellipse(mask, (self.width//2, self.height//2), 
                   (self.width//3, self.height//2), 0, 0, 360, 255, -1)
        return mask
    
    def create_single_window(self):
        """Tạo cửa sổ duy nhất chứa video và controls"""
        # Tạo figure với layout tùy chỉnh
        self.fig = plt.figure(figsize=(12, 10))
        plt.suptitle('Pressure Map ', fontsize=16, fontweight='bold')
        
        # Main plot area cho pressure map - căn giữa
        self.ax_main = plt.axes([0.1, 0.15, 0.8, 0.75])  # [left, bottom, width, height]
        
        # Control panel ở dưới
        ax_play = plt.axes([0.3, 0.05, 0.1, 0.05])
        ax_slider = plt.axes([0.1, 0.02, 0.8, 0.02])
        
        # Tạo controls
        self.btn_play = Button(ax_play, 'Pause', color='lightcoral')
        
        # Đảm bảo slider có giá trị max hợp lệ
        max_frame = max(1, self.total_frames - 1) if self.total_frames > 0 else 1
        self.frame_slider = Slider(ax_slider, 'Frame', 0, max_frame, 
                                  valinit=0, valfmt='%d')
        
        # Connect events
        self.btn_play.on_clicked(self.toggle_play_pause)
        self.frame_slider.on_changed(self.on_slider_change)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Setup colormap
        colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
        self.pressure_cmap = LinearSegmentedColormap.from_list('pressure', colors, N=256)
        
        # Tính max pressure để scale colorbar
        self.max_pressure = max(self.sensor_data.max().max(), 1.0) if self.total_frames > 0 else 1.0
        
        print("\n🎮 ĐIỀU KHIỂN:")
        print("  Space: Play/Pause")
        print("  ←/→: Frame trước/sau")
        print("  Kéo slider: Nhảy frame")
        print(f"  Tổng cộng: {self.total_frames} frames\n")
    
    def convert_coordinates_and_interpolate(self, pressure_values):
        """Chuyển đổi tọa độ và tạo grid áp suất"""
        # Convert sensor positions to normalized coordinates
        sensor_positions = {}
        for sensor_name, (gx, gy) in SENSOR_POSITIONS.items():
            pixel_x = gx * CELL_SIZE + CELL_SIZE // 2
            pixel_y = gy * CELL_SIZE + CELL_SIZE // 2
            
            norm_x = (pixel_x / self.width) * 100
            norm_y = ((self.height - pixel_y) / self.height) * 100
            
            sensor_positions[sensor_name] = [norm_x, norm_y]
        
        # Create pressure grid using IDW interpolation
        pressure_grid = np.zeros((self.rows, self.cols))
        
        sensor_coords = np.array(list(sensor_positions.values()))
        sensor_values = np.array([pressure_values.get(name, 0) for name in sensor_positions.keys()])
        
        for row in range(self.rows):
            for col in range(self.cols):
                center_x = col * CELL_SIZE + CELL_SIZE // 2
                center_y = row * CELL_SIZE + CELL_SIZE // 2
                
                # Check if point is within foot area
                if (center_y < self.height and center_x < self.width and 
                    self.foot_mask[center_y, center_x] > 0):
                    
                    # Normalize target point
                    norm_x = (center_x / self.width) * 100
                    norm_y = ((self.height - center_y) / self.height) * 100
                    target_point = np.array([[norm_x, norm_y]])
                    
                    # Calculate distances to all sensors
                    distances = cdist(target_point, sensor_coords)[0]
                    distances = np.where(distances == 0, 1e-10, distances)
                    
                    # IDW interpolation
                    weights = 1.0 / (distances ** POWER)
                    weights_normalized = weights / weights.sum()
                    
                    pressure_value = np.sum(weights_normalized * sensor_values)
                    pressure_grid[row, col] = pressure_value
                else:
                    pressure_grid[row, col] = np.nan
        
        return pressure_grid, sensor_positions, pressure_values
    
    def update_frame(self, frame_idx=None):
        """Cập nhật frame hiện tại"""
        if frame_idx is not None:
            self.current_frame = int(frame_idx)
        
        if self.total_frames == 0:
            return
        
        # Đảm bảo frame trong phạm vi hợp lệ
        self.current_frame = max(0, min(self.current_frame, self.total_frames - 1))
        
        # Lấy dữ liệu frame hiện tại
        current_data = self.sensor_data.iloc[self.current_frame].to_dict()
        
        # Tạo pressure grid
        pressure_grid, sensor_positions, pressure_values = self.convert_coordinates_and_interpolate(current_data)
        
        # Clear và vẽ lại main plot
        self.ax_main.clear()
        
        # Hiển thị pressure map
        display_grid = np.where(np.isnan(pressure_grid), 0, pressure_grid)
        
        im = self.ax_main.imshow(display_grid, cmap=self.pressure_cmap, 
                                vmin=0, vmax=self.max_pressure, 
                                aspect='equal', interpolation='bilinear')
        
        # Vẽ sensor positions và values
        for sensor_name, (gx, gy) in SENSOR_POSITIONS.items():
            pressure = current_data.get(sensor_name, 0)
            
            # Sensor marker
            self.ax_main.plot(gx, gy, 'o', color='white', markersize=10, 
                             markeredgecolor='black', markeredgewidth=2)
            
            # Sensor label
            self.ax_main.text(gx, gy-3, sensor_name, ha='center', va='center', 
                             fontsize=8, fontweight='bold', color='white',
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.8))
            
            # Pressure value
            self.ax_main.text(gx, gy+3, f'{pressure:.2f}', ha='center', va='center', 
                             fontsize=7, fontweight='bold', color='black',
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.9))
        
        # Setup axes - chỉ hiển thị frame number
        self.ax_main.set_title(f'Frame {self.current_frame + 1}/{self.total_frames}', 
                              fontsize=14, fontweight='bold')
        self.ax_main.set_xlim(-0.5, self.cols-0.5)
        self.ax_main.set_ylim(self.rows-0.5, -0.5)
        self.ax_main.set_xlabel('Grid X')
        self.ax_main.set_ylabel('Grid Y')
        
        # Add colorbar nếu chưa có
        if not hasattr(self, 'colorbar'):
            self.colorbar = self.fig.colorbar(im, ax=self.ax_main, shrink=0.6)
            self.colorbar.set_label('Pressure (Bar)', rotation=270, labelpad=15)
        
        # Update slider position
        if hasattr(self, 'frame_slider'):
            self.frame_slider.set_val(self.current_frame)
        
        self.fig.canvas.draw_idle()
    
    def animate(self, frame):
        """Animation function"""
        if self.is_playing and self.total_frames > 0:
            self.current_frame = frame % self.total_frames
            self.update_frame()
        return []
    
    def toggle_play_pause(self, event):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Play' if not self.is_playing else 'Pause')
        self.fig.canvas.draw_idle()
    
    def on_slider_change(self, val):
        """Xử lý slider change"""
        if not self.is_playing:
            self.current_frame = int(val)
            self.update_frame()
    
    def on_key_press(self, event):
        """Xử lý phím tắt"""
        if event.key == ' ':  # Space
            self.toggle_play_pause(None)
        elif event.key == 'left' and self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame()
        elif event.key == 'right' and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.update_frame()
    
    def start_video(self, interval=100):
        """Bắt đầu phát video"""
        if self.total_frames > 0:
            print(f"🎬 Bắt đầu phát video với {self.total_frames} frames")
            
            # Hiển thị frame đầu tiên
            self.update_frame(0)
            
            # Tạo animation
            self.animation = FuncAnimation(
                self.fig, self.animate, frames=self.total_frames,
                interval=interval, repeat=True, blit=False
            )
            
            plt.show()
        else:
            print("❌ Không có dữ liệu để phát!")

def main():
    """Hàm chính"""
    print("🚀 Khởi động Pressure Map Video Player...")
    print("📁 Tìm kiếm file dữ liệu...")
    
    # Thử các tên file có thể có
    possible_files = [
        "foot_pressure_continuous_data.xls",
        "foot_pressure_continuous_data.xlsx", 
        "foot_pressure_data.csv",
        "Dataa.xlsx",
        "data.xlsx"
    ]
    
    excel_file = None
    for filename in possible_files:
        try:
            # Kiểm tra file tồn tại
            import os
            if os.path.exists(filename):
                excel_file = filename
                print(f"✓ Tìm thấy file: {filename}")
                break
        except:
            continue
    
    if not excel_file:
        print("⚠ Không tìm thấy file dữ liệu, sẽ sử dụng dữ liệu mẫu")
        excel_file = "foot_pressure_continuous_data.xls"  # Sẽ tạo sample data
    
    try:
        # Tạo video player
        player = PressureMapVideoPlayer(excel_file, IMAGE_PATH)
        
        # Bắt đầu phát video (100ms = 10 FPS)
        player.start_video(interval=100)
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()