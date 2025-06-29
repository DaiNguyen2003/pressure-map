import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.colors import LinearSegmentedColormap

# --- Phần 1: Hàm xử lý ảnh để lấy mặt nạ hình dáng bàn chân ---
def get_foot_shape_mask(image_path="feet-outline-paper-crafts.png", 
                        target_width=100, target_height=100, 
                        rotate_left_90=False):
    final_output_mask = np.zeros((target_height, target_width), dtype=np.uint8)

    try:
        image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_original is None:
            raise FileNotFoundError(f"Không thể tải ảnh từ {image_path}")

        _, binary_shape_image = cv2.threshold(image_original, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_shape_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("Cảnh báo: Không tìm thấy đường viền nào trong ảnh.")
            raise ValueError("Không tìm thấy đường viền")

        mask_from_image_contours = np.zeros_like(image_original, dtype=np.uint8)
        cv2.drawContours(mask_from_image_contours, contours, -1, (255), thickness=cv2.FILLED)

        if rotate_left_90:
            print("Đang xoay mặt nạ bàn chân từ ảnh 90 độ ngược chiều kim đồng hồ.")
            mask_from_image_contours = cv2.rotate(mask_from_image_contours, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        final_output_mask = cv2.resize(mask_from_image_contours, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        return final_output_mask

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý ảnh hoặc không tìm thấy tệp: {e}. Sử dụng mặt nạ chữ nhật mặc định (xoay nếu được yêu cầu).")
        
        dummy_canvas_unrotated = np.zeros((target_height, target_width), dtype=np.uint8)
        row_start, row_end = int(0.1 * target_height), int(0.9 * target_height)
        col_start, col_end = int(0.1 * target_width), int(0.9 * target_width)
        cv2.rectangle(dummy_canvas_unrotated, (col_start, row_start), (col_end, row_end), (255), cv2.FILLED)

        if rotate_left_90:
            print("Đang xoay mặt nạ chữ nhật mặc định 90 độ ngược chiều kim đồng hồ.")
            rotated_dummy = cv2.rotate(dummy_canvas_unrotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
            final_output_mask = cv2.resize(rotated_dummy, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        else:
            final_output_mask = dummy_canvas_unrotated
        
        return final_output_mask

# --- Phần 2: Lớp FootPressureIDW đã cập nhật phương thức plot_pressure_map ---
class FootPressureIDW:
    def __init__(self, filled_foot_mask=None):
        self.sensor_positions = {
            'M01': [85, 48], 'M02': [58, 35], 'M03': [58, 70], 'M04': [30, 20],
            'M05': [30, 47], 'M06': [30, 75], 'M07': [12, 25], 'M08': [12, 65]
        }
        self.pressure_values = {
            'M01': 4.43, 'M02': 0.27, 'M03': 0, 'M04': 0.8,
            'M05': 0.17, 'M06': 0.98, 'M07': 2.52, 'M08': 0.07
        }
        self.grid_resolution = 100
        self.x_grid, self.y_grid = np.meshgrid(
            np.linspace(0, 100, self.grid_resolution),
            np.linspace(0, 100, self.grid_resolution)
        )
        self.filled_foot_mask = filled_foot_mask
        
        if self.filled_foot_mask is not None:
            if (self.filled_foot_mask.shape[0] != self.grid_resolution or
                self.filled_foot_mask.shape[1] != self.grid_resolution):
                print(f"Cảnh báo: Hình dạng mặt nạ {self.filled_foot_mask.shape} "
                      f"không khớp grid_resolution ({self.grid_resolution}x{self.grid_resolution}). "
                      "Đang thay đổi kích thước.")
                self.filled_foot_mask = cv2.resize(
                    self.filled_foot_mask,
                    (self.grid_resolution, self.grid_resolution), 
                    interpolation=cv2.INTER_NEAREST
                )

    def is_in_foot_shape_rect(self, x, y):
        if 70 <= x <= 95 and 26 <= y <= 89: return True
        if 42 <= x <= 75 and 24 <= y <= 89: return True
        if 5 <= x <= 45 and 5 <= y <= 85: return True
        return False

    def idw_interpolation(self, target_points, power=2):
        sensor_coords = np.array(list(self.sensor_positions.values()))
        sensor_values = np.array(list(self.pressure_values.values()))
        distances = cdist(target_points, sensor_coords)
        distances = np.where(distances == 0, 1e-10, distances)
        weights = 1.0 / (distances ** power)
        weights_normalized = weights / weights.sum(axis=1, keepdims=True)
        interpolated_values = np.sum(weights_normalized * sensor_values, axis=1)
        return interpolated_values

    def create_pressure_map(self):
        points = np.column_stack([self.x_grid.flatten(), self.y_grid.flatten()])
        interpolated_values = self.idw_interpolation(points, power=2)
        pressure_grid = interpolated_values.reshape(self.x_grid.shape)
        
        current_foot_mask = np.zeros_like(self.x_grid, dtype=bool)

        if self.filled_foot_mask is not None:
            mask_h, mask_w = self.filled_foot_mask.shape

            for i in range(self.grid_resolution):
                for j in range(self.grid_resolution):
                    x_coord_0_100 = self.x_grid[i, j]
                    y_coord_0_100 = self.y_grid[i, j]
                    mask_col = int(x_coord_0_100 / 100.0 * (mask_w - 1))
                    mask_row = int(((100.0 - y_coord_0_100) / 100.0) * (mask_h - 1))
                    mask_col = np.clip(mask_col, 0, mask_w - 1)
                    mask_row = np.clip(mask_row, 0, mask_h - 1)
                    
                    if self.filled_foot_mask[mask_row, mask_col] > 0:
                        current_foot_mask[i, j] = True
        else:
            print("Không có mặt nạ hình ảnh. Sử dụng hình chữ nhật mặc định.")
            for i in range(self.grid_resolution):
                for j in range(self.grid_resolution):
                    if self.is_in_foot_shape_rect(self.x_grid[i,j], self.y_grid[i,j]):
                        current_foot_mask[i,j] = True
        
        return np.where(current_foot_mask, pressure_grid, np.nan)

    def plot_pressure_map(self, figsize=(10, 8)): # Bỏ x_axis_scale_factor nếu không dùng nữa
        """Vẽ bản đồ áp suất 2D với tỷ lệ trục x:y = 2:1."""
        pressure_grid = self.create_pressure_map()
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        #custom_colors = [ '#0000FF','#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
        
      
        custom_colors = ['#000000', '#404040', '#808080', '#bfbfbf', '#ffffff']
        cmap = LinearSegmentedColormap.from_list('pressure_custom', custom_colors, N=256)

        vmin_data = 0.0
        if self.pressure_values:
            current_sensor_readings = list(self.pressure_values.values())
            numeric_readings = [val for val in current_sensor_readings if isinstance(val, (int, float))]
            if numeric_readings:
                calculated_max = np.nanmax(numeric_readings)
                if np.isnan(calculated_max):
                    vmax_data = 4.5
                    print("Cảnh báo: Áp suất tối đa là NaN. Sử dụng vmax=4.5.")
                else:
                    vmax_data = float(calculated_max)
            else:
                vmax_data = 4.5
                print("Cảnh báo: Không có giá trị áp suất số. Sử dụng vmax=4.5.")
        else:
            vmax_data = 4.5
            print("Cảnh báo: Từ điển áp suất rỗng. Sử dụng vmax=4.5.")
        
        # --- CẬP NHẬT CHO TỶ LỆ TRỤC ---
        # Dữ liệu logic X và Y vẫn là 0-100
        plot_extent = [0, 100, 0, 100] 
        
        im = ax.imshow(pressure_grid, extent=plot_extent, 
                       cmap=cmap, origin='lower', vmin=vmin_data, vmax=vmax_data)
        
        # Đường đồng mức: sử dụng self.x_grid và self.y_grid gốc (0-100)
        if np.any(~np.isnan(pressure_grid)):
             contours = ax.contour(self.x_grid, self.y_grid, np.nan_to_num(pressure_grid),
                                   levels=10, colors='black', alpha=0.3, linewidths=0.7)
             ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        
        # Vẽ vị trí và giá trị cảm biến: sử dụng tọa độ X gốc (0-100)
        for sensor, pos in self.sensor_positions.items():
            ax.plot(pos[0], pos[1], 'o', markersize=8, markerfacecolor='white', 
                    markeredgecolor='black', markeredgewidth=1.5)
            ax.text(pos[0], pos[1] - 5, f'{self.pressure_values[sensor]:.1f}N', 
                    ha='center', va='top', fontsize=7, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
        
        # Thêm nhãn vùng: sử dụng tọa độ X gốc (0-100)
        ax.text(25, 92, 'Forefoot', fontsize=11, fontweight='bold', color='dimgray', ha='center')
        ax.text(58, 92, 'Midfoot', fontsize=11, fontweight='bold', color='dimgray', ha='center')
        ax.text(80, 92, 'Heel/Rearfoot', fontsize=11, fontweight='bold', color='dimgray', ha='center')

        ax.set_title('Bản đồ áp suất lòng bàn chân - IDW (Mặt nạ ảnh)', 
                     fontsize=15, fontweight='bold')
        ax.set_xlabel('Tọa độ X', fontsize=11)
        ax.set_ylabel('Tọa độ Y', fontsize=11)
        
        # Đặt giới hạn trục X và Y là 0-100
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        ax.grid(True, linestyle=':', alpha=0.4)
        
        # Đặt tỷ lệ khung hình: Yunit / Xunit = 0.5, nghĩa là Xunit dài gấp đôi Yunit
        # Điều này sẽ làm cho hộp chứa dữ liệu (0-100 X và 0-100 Y) có tỷ lệ rộng:cao là 2:1.
        ax.set_aspect(0.45, adjustable='box') 

        cbar = plt.colorbar(im, ax=ax, shrink=0.75, aspect=15)
        cbar.set_label('Áp suất (N)', rotation=270, labelpad=18, fontsize=11)
        
        plt.tight_layout()
        return fig
        # --- KẾT THÚC CẬP NHẬT CHO TỶ LỆ TRỤC ---

# --- Phần 3: Khối thực thi chính ---
if __name__ == "__main__":
    foot_image_file = "feet-outline-paper-crafts.png" 
    
    print(f"Đang tải mặt nạ hình dáng bàn chân từ: {foot_image_file}")
    foot_mask = get_foot_shape_mask(
        image_path=foot_image_file,
        target_width=100, 
        target_height=100,
        rotate_left_90=True # Giữ logic xoay nếu bạn vẫn muốn xoay mặt nạ
    )

    print("Đang khởi tạo bản đồ áp suất với mặt nạ bàn chân đã xử lý...")
    foot_pressure_analyzer = FootPressureIDW(filled_foot_mask=foot_mask)
    
    print("Đang tạo và vẽ bản đồ áp suất với tỷ lệ trục x:y = 2:1...")
    # Gọi plot_pressure_map không cần x_axis_scale_factor, vì tỷ lệ được xử lý bằng set_aspect
    pressure_map_figure = foot_pressure_analyzer.plot_pressure_map()
    plt.show()