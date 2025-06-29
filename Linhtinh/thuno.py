import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SensorData:
    """Lớp dữ liệu cho thông tin cảm biến"""
    position: Tuple[float, float]
    pressure: float
    name: str

class FootShapeProcessor:
    """Lớp xử lý hình dạng bàn chân từ ảnh"""
    
    def __init__(self, target_size: Tuple[int, int] = (100, 100)):
        self.target_width, self.target_height = target_size
        
    def load_and_process_image(self, image_path: Union[str, Path], 
                             rotate_left_90: bool = False) -> np.ndarray:
        """
        Tải và xử lý ảnh để tạo mặt nạ hình dáng bàn chân
        
        Args:
            image_path: Đường dẫn tới file ảnh
            rotate_left_90: Có xoay 90 độ ngược chiều kim đồng hồ không
            
        Returns:
            Mặt nạ nhị phân của hình dáng bàn chân
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file: {image_path}")
                
            # Đọc ảnh grayscale
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh từ: {image_path}")
                
            logger.info(f"Đã tải ảnh: {image_path} với kích thước {image.shape}")
            
            # Tạo ảnh nhị phân
            mask = self._create_binary_mask(image)
            
            # Xoay nếu cần
            if rotate_left_90:
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                logger.info("Đã xoay mặt nạ 90° ngược chiều kim đồng hồ")
            
            # Resize về kích thước mục tiêu
            final_mask = cv2.resize(mask, (self.target_width, self.target_height), 
                                  interpolation=cv2.INTER_NEAREST)
            
            return final_mask
            
        except Exception as e:
            logger.warning(f"Lỗi xử lý ảnh: {e}. Sử dụng mặt nạ mặc định")
            return self._create_default_mask(rotate_left_90)
    
    def _create_binary_mask(self, image: np.ndarray) -> np.ndarray:
        """Tạo mặt nạ nhị phân từ ảnh grayscale"""
        # Áp dụng threshold
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("Không tìm thấy đường viền trong ảnh")
        
        # Tạo mặt nạ từ contours
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        
        # Làm mịn mặt nạ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _create_default_mask(self, rotate_left_90: bool = False) -> np.ndarray:
        """Tạo mặt nạ mặc định hình chữ nhật"""
        mask = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
        
        # Tạo hình chữ nhật với padding 10%
        row_start, row_end = int(0.1 * self.target_height), int(0.9 * self.target_height)
        col_start, col_end = int(0.1 * self.target_width), int(0.9 * self.target_width)
        cv2.rectangle(mask, (col_start, row_start), (col_end, row_end), 255, cv2.FILLED)
        
        if rotate_left_90:
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask = cv2.resize(mask, (self.target_width, self.target_height), 
                            interpolation=cv2.INTER_NEAREST)
        
        logger.info("Đã tạo mặt nạ mặc định")
        return mask

class FootPressureAnalyzer:
    """Lớp phân tích áp suất bàn chân sử dụng IDW interpolation"""
    
    # Vùng bàn chân mặc định
    DEFAULT_FOOT_REGIONS = [
        (70, 95, 26, 89),  # Heel/Rearfoot
        (42, 75, 24, 89),  # Midfoot  
        (5, 45, 5, 85)     # Forefoot
    ]
    
    # Màu sắc mặc định cho bản đồ nhiệt
    #DEFAULT_COLORS = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
    DEFAULT_COLORS = ['#000000', '#404040', '#808080', '#bfbfbf', '#ffffff']
    def __init__(self, sensors: Optional[Dict[str, SensorData]] = None,
                 foot_mask: Optional[np.ndarray] = None,
                 grid_resolution: int = 100):
        """
        Khởi tạo bộ phân tích áp suất bàn chân
        
        Args:
            sensors: Dictionary chứa dữ liệu cảm biến
            foot_mask: Mặt nạ hình dáng bàn chân
            grid_resolution: Độ phân giải grid interpolation
        """
        self.grid_resolution = grid_resolution
        self.sensors = sensors or self._get_default_sensors()
        self.foot_mask = self._validate_and_resize_mask(foot_mask)
        
        # Tạo grid tọa độ
        self.x_grid, self.y_grid = np.meshgrid(
            np.linspace(0, 100, self.grid_resolution),
            np.linspace(0, 100, self.grid_resolution)
        )
        
        logger.info(f"Khởi tạo analyzer với {len(self.sensors)} cảm biến, "
                   f"grid {self.grid_resolution}x{self.grid_resolution}")
    
    def _get_default_sensors(self) -> Dict[str, SensorData]:
        """Tạo dữ liệu cảm biến mặc định"""
        default_data = {
            'M01': ([85, 48], 4.43), 'M02': ([58, 35], 0.27), 'M03': ([58, 70], 0.0),
            'M04': ([30, 20], 0.8), 'M05': ([30, 47], 0.17), 'M06': ([30, 75], 0.98),
            'M07': ([12, 25], 2.52), 'M08': ([12, 65], 0.07)
        }
        
        return {name: SensorData(tuple(pos), pressure, name) 
                for name, (pos, pressure) in default_data.items()}
    
    def _validate_and_resize_mask(self, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Kiểm tra và thay đổi kích thước mặt nạ nếu cần"""
        if mask is None:
            return None
            
        target_shape = (self.grid_resolution, self.grid_resolution)
        if mask.shape != target_shape:
            logger.info(f"Thay đổi kích thước mặt nạ từ {mask.shape} về {target_shape}")
            return cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def update_sensor_data(self, sensor_name: str, position: Tuple[float, float], 
                          pressure: float) -> None:
        """Cập nhật dữ liệu cảm biến"""
        self.sensors[sensor_name] = SensorData(position, pressure, sensor_name)
        logger.info(f"Đã cập nhật cảm biến {sensor_name}: {position}, {pressure}N")
    
    def _check_point_in_foot_regions(self, x: float, y: float) -> bool:
        """Kiểm tra điểm có nằm trong vùng bàn chân không (dự phòng)"""
        return any(x_min <= x <= x_max and y_min <= y <= y_max 
                  for x_min, x_max, y_min, y_max in self.DEFAULT_FOOT_REGIONS)
    
    def idw_interpolation(self, target_points: np.ndarray, power: float = 2.0) -> np.ndarray:
        """
        Thực hiện IDW interpolation
        
        Args:
            target_points: Các điểm cần interpolate (N x 2)
            power: Lũy thừa cho IDW (thường là 2)
            
        Returns:
            Mảng giá trị áp suất được interpolate
        """
        if not self.sensors:
            return np.zeros(len(target_points))
        
        # Lấy tọa độ và giá trị cảm biến
        sensor_coords = np.array([sensor.position for sensor in self.sensors.values()])
        sensor_values = np.array([sensor.pressure for sensor in self.sensors.values()])
        
        # Tính khoảng cách
        distances = cdist(target_points, sensor_coords)
        
        # Tránh chia cho 0
        distances = np.where(distances < 1e-10, 1e-10, distances)
        
        # Tính trọng số IDW
        weights = 1.0 / (distances ** power)
        weights_sum = weights.sum(axis=1, keepdims=True)
        
        # Tránh chia cho 0 khi tất cả trọng số là 0
        weights_sum = np.where(weights_sum == 0, 1e-10, weights_sum)
        weights_normalized = weights / weights_sum
        
        # Interpolation
        interpolated_values = np.sum(weights_normalized * sensor_values, axis=1)
        return interpolated_values
    
    def create_pressure_map(self, power: float = 2.0) -> np.ndarray:
        """
        Tạo bản đồ áp suất 2D
        
        Args:
            power: Lũy thừa cho IDW interpolation
            
        Returns:
            Grid áp suất 2D với NaN ở vùng ngoài bàn chân
        """
        # Tạo các điểm grid
        points = np.column_stack([self.x_grid.flatten(), self.y_grid.flatten()])
        
        # Interpolation
        interpolated_values = self.idw_interpolation(points, power)
        pressure_grid = interpolated_values.reshape(self.x_grid.shape)
        
        # Tạo mặt nạ vùng bàn chân
        foot_region_mask = self._create_foot_region_mask()
        
        # Áp dụng mặt nạ
        return np.where(foot_region_mask, pressure_grid, np.nan)
    
    def _create_foot_region_mask(self) -> np.ndarray:
        """Tạo mặt nạ vùng bàn chân"""
        mask = np.zeros_like(self.x_grid, dtype=bool)
        
        if self.foot_mask is not None:
            # Sử dụng mặt nạ từ ảnh
            mask_h, mask_w = self.foot_mask.shape
            
            for i in range(self.grid_resolution):
                for j in range(self.grid_resolution):
                    x_coord = self.x_grid[i, j]
                    y_coord = self.y_grid[i, j]
                    
                    # Chuyển đổi tọa độ grid sang tọa độ mặt nạ
                    mask_col = int(x_coord / 100.0 * (mask_w - 1))
                    mask_row = int((100.0 - y_coord) / 100.0 * (mask_h - 1))
                    
                    # Clamp về giới hạn
                    mask_col = np.clip(mask_col, 0, mask_w - 1)
                    mask_row = np.clip(mask_row, 0, mask_h - 1)
                    
                    if self.foot_mask[mask_row, mask_col] > 0:
                        mask[i, j] = True
        else:
            # Sử dụng vùng hình chữ nhật mặc định
            logger.info("Sử dụng vùng bàn chân mặc định")
            for i in range(self.grid_resolution):
                for j in range(self.grid_resolution):
                    x_coord = self.x_grid[i, j]
                    y_coord = self.y_grid[i, j]
                    if self._check_point_in_foot_regions(x_coord, y_coord):
                        mask[i, j] = True
        
        return mask
    
    def plot_pressure_map(self, figsize: Tuple[int, int] = (12, 8),
                         power: float = 2.0,
                         colors: Optional[List[str]] = None,
                         show_contours: bool = True,
                         show_sensors: bool = True,
                         title: Optional[str] = None) -> plt.Figure:
        """
        Vẽ bản đồ áp suất với nhiều tùy chọn
        
        Args:
            figsize: Kích thước figure
            power: Lũy thừa IDW
            colors: Danh sách màu cho colormap
            show_contours: Hiển thị đường đồng mức
            show_sensors: Hiển thị vị trí cảm biến
            title: Tiêu đề tùy chỉnh
            
        Returns:
            Figure matplotlib
        """
        # Tạo bản đồ áp suất
        pressure_grid = self.create_pressure_map(power)
        
        # Tính toán giới hạn màu
        pressure_values = [s.pressure for s in self.sensors.values()]
        vmin, vmax = 0.0, max(pressure_values) if pressure_values else 4.5
        
        # Tạo figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Tạo colormap
        colors = colors or self.DEFAULT_COLORS
        cmap = LinearSegmentedColormap.from_list('pressure_custom', colors, N=256)
        
        # Vẽ heatmap
        extent = [0, 100, 0, 100]
        im = ax.imshow(pressure_grid, extent=extent, cmap=cmap, 
                      origin='lower', vmin=vmin, vmax=vmax)
        
        # Vẽ đường đồng mức
        if show_contours and np.any(~np.isnan(pressure_grid)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                contours = ax.contour(self.x_grid, self.y_grid, 
                                    np.nan_to_num(pressure_grid),
                                    levels=8, colors='black', alpha=0.4, linewidths=0.8)
                ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        
        # Vẽ cảm biến
        if show_sensors:
            self._plot_sensors(ax)
        
        # Thêm nhãn vùng
        self._add_region_labels(ax)
        
        # Cấu hình axes
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect(0.45, adjustable='box')  # Tỷ lệ 2:1
        ax.grid(True, linestyle=':', alpha=0.3)
        
        # Tiêu đề và nhãn
        default_title = f'Bản đồ áp suất lòng bàn chân'
        ax.set_title(title or default_title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Tọa độ X (mm)', fontsize=12)
        ax.set_ylabel('Tọa độ Y (mm)', fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Áp suất (N)', rotation=270, labelpad=20, fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def _plot_sensors(self, ax) -> None:
        """Vẽ vị trí và giá trị cảm biến"""
        for sensor in self.sensors.values():
            x, y = sensor.position
            
            # Vẽ điểm cảm biến
            ax.plot(x, y, 'o', markersize=10, markerfacecolor='white', 
                   markeredgecolor='black', markeredgewidth=2)
            
            # Thêm nhãn giá trị
            ax.text(x, y - 4, f'{sensor.pressure:.1f}N', 
                   ha='center', va='top', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           alpha=0.8, edgecolor='gray'))
            
            # Thêm tên cảm biến
            ax.text(x, y + 4, sensor.name, ha='center', va='bottom', 
                   fontsize=7, color='darkblue', fontweight='bold')
    
    def _add_region_labels(self, ax) -> None:
        """Thêm nhãn vùng bàn chân"""
        regions = [
            (25, 95, 'Forefoot', 'dimgray'),
            (58, 95, 'Midfoot', 'dimgray'), 
            (80, 95, 'Heel/Rearfoot', 'dimgray')
        ]
        
        for x, y, label, color in regions:
            ax.text(x, y, label, fontsize=12, fontweight='bold', 
                   color=color, ha='center', va='bottom')
    
    def get_pressure_statistics(self) -> Dict[str, float]:
        """Tính toán thống kê áp suất"""
        pressure_values = [s.pressure for s in self.sensors.values()]
        
        if not pressure_values:
            return {}
        
        return {
            'min': min(pressure_values),
            'max': max(pressure_values),
            'mean': np.mean(pressure_values),
            'std': np.std(pressure_values),
            'total': sum(pressure_values)
        }
    
    def export_data(self, filename: str) -> None:
        """Xuất dữ liệu ra file CSV"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Sensor', 'X', 'Y', 'Pressure'])
            
            for sensor in self.sensors.values():
                writer.writerow([sensor.name, sensor.position[0], 
                               sensor.position[1], sensor.pressure])
        
        logger.info(f"Đã xuất dữ liệu ra {filename}")

def main():
    """Hàm chính để chạy ví dụ"""
    # Đường dẫn ảnh
    image_path = "feet-outline-paper-crafts.png"
    
    # Xử lý hình dáng bàn chân
    shape_processor = FootShapeProcessor(target_size=(100, 100))
    foot_mask = shape_processor.load_and_process_image(
        image_path, rotate_left_90=True
    )
    
    # Tạo analyzer
    analyzer = FootPressureAnalyzer(foot_mask=foot_mask, grid_resolution=100)
    
    # In thống kê
    stats = analyzer.get_pressure_statistics()
    logger.info(f"Thống kê áp suất: {stats}")
    
    # Vẽ bản đồ áp suất
    fig = analyzer.plot_pressure_map(
        figsize=(14, 8),
        power=2.0,
        show_contours=True,
        show_sensors=True,
        title="Bản đồ áp suất bàn chân - Phiên bản tối ưu"
    )
    
    # Xuất dữ liệu
    analyzer.export_data("sensor_data.csv")
    
    plt.show()

if __name__ == "__main__":
    main()