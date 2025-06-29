import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
image_path = "DOan/Image/banchan.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
thicker_contour_image = np.zeros_like(image)
cv2.drawContours(thicker_contour_image, contours, -1, (255), thickness=5)
cell_size = 20  
grid_image = cv2.cvtColor(thicker_contour_image, cv2.COLOR_GRAY2RGB)
height, width = grid_image.shape[:2]
cols = width // cell_size
rows = height // cell_size
fig, ax = plt.subplots(figsize=(12, 16))
ax.imshow(grid_image)
for y in range(0, height, cell_size):
    ax.axhline(y=y, color='green', linewidth=1, alpha=0.7)
for x in range(0, width, cell_size):
    ax.axvline(x=x, color='green', linewidth=1, alpha=0.7)

coord_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                    verticalalignment='top', fontsize=12, fontweight='bold')


highlight_rect = Rectangle((0, 0), cell_size, cell_size, 
                          linewidth=3, edgecolor='red', facecolor='red', alpha=0.3)
ax.add_patch(highlight_rect)
highlight_rect.set_visible(False)

def on_mouse_move(event):
    """Xử lý sự kiện di chuyển chuột"""
    if event.inaxes != ax:
        highlight_rect.set_visible(False)
        coord_text.set_text('')
        fig.canvas.draw_idle()
        return
    mouse_x, mouse_y = event.xdata, event.ydata
    if mouse_x is None or mouse_y is None:
        return
    grid_x = int(mouse_x // cell_size)
    grid_y = int(mouse_y // cell_size)
    if 0 <= grid_x < cols and 0 <= grid_y < rows:
        highlight_rect.set_xy((grid_x * cell_size, grid_y * cell_size))
        highlight_rect.set_visible(True)
        coord_text.set_text(f'Grid: ({grid_x}, {grid_y})')
        fig.canvas.draw_idle()
    else:
        highlight_rect.set_visible(False)
        coord_text.set_text('')
        fig.canvas.draw_idle()
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
ax.set_xticks([])
ax.set_yticks([])
print(f"Grid created: {cols} columns x {rows} rows")
print(f"Each cell: {cell_size}x{cell_size} pixels")
print(f"Total cells: {cols * rows}")
print("Move your mouse over the grid to see grid coordinates!")
plt.tight_layout()
plt.show()