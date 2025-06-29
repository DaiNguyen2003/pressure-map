import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import serial
import serial.tools.list_ports
import threading
import time
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque

class ArduinoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arduino Data Reader with Graph")
        self.root.geometry("1200x800")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Serial connection variables
        self.serial_port = None
        self.is_connected = False
        self.is_running = False
        self.thread = None
        
        # Data storage for graphing (8 values)
        self.max_data_points = 100  # Show last 100 data points
        self.data_buffers = [deque(maxlen=self.max_data_points) for _ in range(8)]
        self.time_buffer = deque(maxlen=self.max_data_points)
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        self.value_labels = [f'Value {i+1}' for i in range(8)]
        
        # Graph display settings
        self.time_window = 30  # Default: show last 30 seconds
        self.use_rolling_window = True  # Use rolling time window
        
        # Create main container
        self.main_container = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls and data display
        self.left_panel = ttk.Frame(self.main_container)
        self.main_container.add(self.left_panel, weight=1)
        
        # Right panel for graph
        self.right_panel = ttk.Frame(self.main_container)
        self.main_container.add(self.right_panel, weight=2)
        
        self.setup_left_panel()
        self.setup_graph_panel()
        
        # Initialize port list
        self.refresh_ports()
        
        # Data counter
        self.data_count = 0
        
        # Raw data storage for saving
        self.raw_data = []
        
    def setup_left_panel(self):
        # Control panel frame
        self.control_frame = ttk.LabelFrame(self.left_panel, text="Controls")
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # COM Port selection
        ttk.Label(self.control_frame, text="COM Port:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(self.control_frame, textvariable=self.port_var, width=10)
        self.port_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Baud rate selection
        ttk.Label(self.control_frame, text="Baud Rate:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.baud_var = tk.StringVar(value="9600")
        self.baud_combo = ttk.Combobox(self.control_frame, textvariable=self.baud_var, 
                                      values=["4800","9600","14400", "19200", "38400", "57600", "115200"], width=8)
        self.baud_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Refresh and connect buttons
        self.refresh_button = ttk.Button(self.control_frame, text="Refresh", command=self.refresh_ports)
        self.refresh_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.connect_button = ttk.Button(self.control_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Start and stop buttons
        self.start_button = ttk.Button(self.control_frame, text="Start", 
                                      state=tk.DISABLED, command=self.start_reading)
        self.start_button.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        
        self.stop_button = ttk.Button(self.control_frame, text="Stop", 
                                     state=tk.DISABLED, command=self.stop_reading)
        self.stop_button.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        
        # Clear and save buttons
        self.clear_button = ttk.Button(self.control_frame, text="Clear", 
                                      command=self.clear_data, state=tk.DISABLED)
        self.clear_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew", columnspan=2)
        
        self.save_button = ttk.Button(self.control_frame, text="Save", 
                                     command=self.save_data, state=tk.DISABLED)
        self.save_button.grid(row=2, column=2, padx=5, pady=5, sticky="ew", columnspan=2)
        
        # Data display area
        self.data_frame = ttk.LabelFrame(self.left_panel, text="Arduino Data")
        self.data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text widget to display data
        text_frame = ttk.Frame(self.data_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.data_text = tk.Text(text_frame, wrap=tk.WORD, width=40, height=15)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.data_text.yview)
        self.data_text.configure(yscrollcommand=scrollbar.set)
        
        self.data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status and counter frame
        self.status_frame = ttk.Frame(self.left_panel)
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Data counter
        ttk.Label(self.status_frame, text="Count:").pack(side=tk.LEFT, padx=5)
        self.count_var = tk.StringVar(value="0")
        ttk.Label(self.status_frame, textvariable=self.count_var).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Not connected")
        self.status_bar = ttk.Label(self.left_panel, textvariable=self.status_var, 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
    def setup_graph_panel(self):
        # Graph panel
        self.graph_frame = ttk.LabelFrame(self.right_panel, text="Real-time Graph (8 Values)")
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("Arduino Data - Real-time Graph")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize empty lines for each value
        self.lines = []
        for i in range(8):
            line, = self.ax.plot([], [], color=self.colors[i], label=self.value_labels[i], linewidth=2)
            self.lines.append(line)
        
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Graph control frame
        self.graph_control_frame = ttk.Frame(self.graph_frame)
        self.graph_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Value selection frame
        self.value_selection_frame = ttk.LabelFrame(self.graph_control_frame, text="Select Values to Display")
        self.value_selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create checkboxes for each value
        self.value_checkboxes = []
        self.value_checkbox_vars = []
        
        # First row of checkboxes (Values 1-4)
        checkbox_row1 = ttk.Frame(self.value_selection_frame)
        checkbox_row1.pack(fill=tk.X, padx=5, pady=2)
        
        for i in range(4):
            var = tk.BooleanVar(value=True)  # All selected by default
            self.value_checkbox_vars.append(var)
            cb = ttk.Checkbutton(checkbox_row1, text=f"Value {i+1}", 
                               variable=var, command=self.update_graph_visibility)
            cb.pack(side=tk.LEFT, padx=10)
            self.value_checkboxes.append(cb)
        
        # Second row of checkboxes (Values 5-8)
        checkbox_row2 = ttk.Frame(self.value_selection_frame)
        checkbox_row2.pack(fill=tk.X, padx=5, pady=2)
        
        for i in range(4, 8):
            var = tk.BooleanVar(value=True)  # All selected by default
            self.value_checkbox_vars.append(var)
            cb = ttk.Checkbutton(checkbox_row2, text=f"Value {i+1}", 
                               variable=var, command=self.update_graph_visibility)
            cb.pack(side=tk.LEFT, padx=10)
            self.value_checkboxes.append(cb)
        
        # Control buttons for selection
        selection_buttons_frame = ttk.Frame(self.value_selection_frame)
        selection_buttons_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(selection_buttons_frame, text="Select All", 
                  command=self.select_all_values).pack(side=tk.LEFT, padx=5)
        ttk.Button(selection_buttons_frame, text="Deselect All", 
                  command=self.deselect_all_values).pack(side=tk.LEFT, padx=5)
        ttk.Button(selection_buttons_frame, text="Toggle All", 
                  command=self.toggle_all_values).pack(side=tk.LEFT, padx=5)
        
        # Graph settings frame
        self.graph_settings_frame = ttk.LabelFrame(self.graph_control_frame, text="Graph Settings")
        self.graph_settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Time window settings
        time_settings_row = ttk.Frame(self.graph_settings_frame)
        time_settings_row.pack(fill=tk.X, padx=5, pady=2)
        
        # Rolling window checkbox
        self.rolling_window_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(time_settings_row, text="Rolling Window", 
                       variable=self.rolling_window_var,
                       command=self.toggle_rolling_window).pack(side=tk.LEFT, padx=5)
        
        # Time window duration
        ttk.Label(time_settings_row, text="Window (sec):").pack(side=tk.LEFT, padx=5)
        self.time_window_var = tk.StringVar(value="30")
        time_window_entry = ttk.Entry(time_settings_row, textvariable=self.time_window_var, width=6)
        time_window_entry.pack(side=tk.LEFT, padx=2)
        time_window_entry.bind('<Return>', self.update_time_window)
        
        ttk.Button(time_settings_row, text="Apply", 
                  command=self.update_time_window).pack(side=tk.LEFT, padx=2)
        
        # Max data points setting
        ttk.Label(time_settings_row, text="Max Points:").pack(side=tk.LEFT, padx=5)
        self.max_points_var = tk.StringVar(value="100")
        max_points_entry = ttk.Entry(time_settings_row, textvariable=self.max_points_var, width=6)
        max_points_entry.pack(side=tk.LEFT, padx=2)
        max_points_entry.bind('<Return>', self.update_max_points)
        
        ttk.Button(time_settings_row, text="Apply", 
                  command=self.update_max_points).pack(side=tk.LEFT, padx=2)
        
        # Auto-scale settings
        settings_row1 = ttk.Frame(self.graph_settings_frame)
        settings_row1.pack(fill=tk.X, padx=5, pady=2)
        
        self.auto_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_row1, text="Auto Scale Y", 
                       variable=self.auto_scale_var).pack(side=tk.LEFT, padx=5)
        
        # Y-axis range controls
        settings_row2 = ttk.Frame(self.graph_settings_frame)
        settings_row2.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(settings_row2, text="Y Min:").pack(side=tk.LEFT, padx=5)
        self.y_min_var = tk.StringVar(value="0")
        ttk.Entry(settings_row2, textvariable=self.y_min_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(settings_row2, text="Y Max:").pack(side=tk.LEFT, padx=5)
        self.y_max_var = tk.StringVar(value="1023")
        ttk.Entry(settings_row2, textvariable=self.y_max_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(settings_row2, text="Apply Range", 
                  command=self.apply_y_range).pack(side=tk.LEFT, padx=5)
        
        # Save image button
        self.save_image_button = ttk.Button(self.graph_settings_frame, text="Save Image", 
                                           command=self.save_graph_image)
        self.save_image_button.pack(side=tk.LEFT, padx=5)
        
    def refresh_ports(self):
        """Refresh available COM ports"""
        ports = serial.tools.list_ports.comports()
        available_ports = [port.device for port in ports]
        self.port_combo["values"] = available_ports
        
        if available_ports:
            self.port_var.set(available_ports[0])
        else:
            self.port_var.set("")
            messagebox.showinfo("Info", "No COM ports found.")
    
    def toggle_connection(self):
        """Connect to or disconnect from the selected COM port"""
        if not self.is_connected:
            self.connect_to_port()
        else:
            self.disconnect_from_port()
    
    def connect_to_port(self):
        """Establish a connection with the Arduino"""
        port = self.port_var.get()
        baud_rate = int(self.baud_var.get())
        
        if not port:
            messagebox.showerror("Error", "Please select a COM port.")
            return
        
        try:
            self.serial_port = serial.Serial(port, baud_rate, timeout=1)
            self.is_connected = True
            self.connect_button.config(text="Disconnect")
            self.start_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.status_var.set(f"Connected to {port} at {baud_rate} baud")
            
            # Clear any data in the buffer
            self.flush_serial_buffer()
        except serial.SerialException as e:
            messagebox.showerror("Connection Error", str(e))
    
    def disconnect_from_port(self):
        """Disconnect from the Arduino"""
        if self.is_running:
            self.stop_reading()
            
        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None
            
        self.is_connected = False
        self.connect_button.config(text="Connect")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.status_var.set("Not connected")
    
    def flush_serial_buffer(self):
        """Clear any data in the serial buffer"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            time.sleep(0.1)
    
    def start_reading(self):
        """Start reading data from Arduino"""
        if not self.is_connected:
            return
        
        self.flush_serial_buffer()
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Reading data...")
        
        # Start the data reading thread
        self.thread = threading.Thread(target=self.read_serial, daemon=True)
        self.thread.start()
        
        # Start the graph update timer
        self.update_graph()
    
    def stop_reading(self):
        """Stop reading data"""
        self.is_running = False
        time.sleep(0.2)
        self.flush_serial_buffer()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set(f"Connected to {self.port_var.get()} - Reading stopped")
    
    def read_serial(self):
        """Read data from the serial port in a separate thread"""
        start_time = time.time()
        
        while self.is_running and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    line_bytes = self.serial_port.readline()
                    line = line_bytes.decode('utf-8', errors='replace').strip()
                    
                    if line:
                        timestamp = time.strftime("%H:%M:%S", time.localtime())
                        current_time = time.time() - start_time
                        
                        # Try to parse 8 comma-separated values
                        try:
                            values = [float(x.strip()) for x in line.split(',')]
                            if len(values) == 8:
                                # Store data for graphing
                                self.time_buffer.append(current_time)
                                for i, value in enumerate(values):
                                    self.data_buffers[i].append(value)
                                
                                # Store raw data for saving
                                self.raw_data.append((timestamp, line))
                                
                                # Update UI in the main thread
                                display_text = f"[{timestamp}] {line}\n"
                                self.root.after(0, lambda text=display_text: self.update_display(text))
                            else:
                                # If not 8 values, still display as text
                                display_text = f"[{timestamp}] {line} (Not 8 values)\n"
                                self.root.after(0, lambda text=display_text: self.update_display(text))
                        except ValueError:
                            # If can't parse as numbers, display as text
                            display_text = f"[{timestamp}] {line} (Parse error)\n"
                            self.root.after(0, lambda text=display_text: self.update_display(text))
                            
            except serial.SerialException as e:
                self.is_running = False
                self.root.after(0, lambda: self.handle_disconnect_error(str(e)))
                break
            except Exception as e:
                self.is_running = False
                self.root.after(0, lambda: self.handle_disconnect_error(f"Error: {str(e)}"))
                break
            
            time.sleep(0.01)
    
    def update_display(self, text):
        """Update the display with new data"""
        self.data_text.insert(tk.END, text)
        self.data_text.see(tk.END)
        
        self.data_count += 1
        self.count_var.set(str(self.data_count))
    
    def update_graph(self):
        """Update the graph with new data"""
        if self.is_running and len(self.time_buffer) > 0:
            # Update each line with new data, but only if it's selected for display
            for i, line in enumerate(self.lines):
                if len(self.data_buffers[i]) > 0:
                    line.set_data(list(self.time_buffer), list(self.data_buffers[i]))
                    # Set visibility based on checkbox state
                    line.set_visible(self.value_checkbox_vars[i].get())
            
            # Update axis limits based on rolling window setting
            if len(self.time_buffer) > 1:
                if self.rolling_window_var.get():
                    # Rolling window: show last N seconds
                    current_time = list(self.time_buffer)[-1]
                    self.ax.set_xlim(max(0, current_time - self.time_window), current_time + 1)
                else:
                    # Fixed window: show from 0 to current time
                    current_time = list(self.time_buffer)[-1]
                    self.ax.set_xlim(0, current_time + 1)
            
            # Auto-scale Y axis or use manual range
            if self.auto_scale_var.get():
                # Only consider visible lines for auto-scaling
                all_values = []
                for i, buffer in enumerate(self.data_buffers):
                    if len(buffer) > 0 and self.value_checkbox_vars[i].get():
                        all_values.extend(list(buffer))
                if all_values:
                    y_min, y_max = min(all_values), max(all_values)
                    margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                    self.ax.set_ylim(y_min - margin, y_max + margin)
            else:
                try:
                    y_min = float(self.y_min_var.get())
                    y_max = float(self.y_max_var.get())
                    self.ax.set_ylim(y_min, y_max)
                except ValueError:
                    pass
            
            self.canvas.draw_idle()
        
        # Schedule next update
        if self.is_running:
            self.root.after(100, self.update_graph)  # Update every 100ms
    
    def toggle_rolling_window(self):
        """Toggle between rolling window and fixed window mode"""
        self.use_rolling_window = self.rolling_window_var.get()
        if not self.use_rolling_window:
            # When switching to fixed window, show all data from 0
            if len(self.time_buffer) > 0:
                current_time = list(self.time_buffer)[-1]
                self.ax.set_xlim(0, current_time + 1)
                self.canvas.draw()
    
    def update_time_window(self, event=None):
        """Update the time window duration"""
        try:
            self.time_window = float(self.time_window_var.get())
            if self.time_window <= 0:
                self.time_window = 30
                self.time_window_var.set("30")
        except ValueError:
            self.time_window = 30
            self.time_window_var.set("30")
            messagebox.showerror("Error", "Please enter a valid positive number for time window.")
    
    def update_max_points(self, event=None):
        """Update the maximum number of data points to store"""
        try:
            new_max = int(self.max_points_var.get())
            if new_max <= 0:
                new_max = 100
                self.max_points_var.set("100")
            
            # Update the maxlen of all deques
            self.max_data_points = new_max
            
            # Create new deques with the new maxlen, preserving existing data
            old_time_data = list(self.time_buffer)
            old_data_buffers = [list(buffer) for buffer in self.data_buffers]
            
            self.time_buffer = deque(old_time_data[-new_max:], maxlen=new_max)
            self.data_buffers = [deque(old_data[-new_max:], maxlen=new_max) 
                               for old_data in old_data_buffers]
            
        except ValueError:
            self.max_data_points = 100
            self.max_points_var.set("100")
            messagebox.showerror("Error", "Please enter a valid positive integer for max points.")
    
    def update_graph_visibility(self):
        """Update the visibility of graph lines based on checkbox selection"""
        for i, line in enumerate(self.lines):
            line.set_visible(self.value_checkbox_vars[i].get())
        
        # Update legend to show only visible lines
        visible_lines = []
        visible_labels = []
        for i, line in enumerate(self.lines):
            if line.get_visible():
                visible_lines.append(line)
                visible_labels.append(self.value_labels[i])
        
        if visible_lines:
            self.ax.legend(visible_lines, visible_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            self.ax.legend().set_visible(False)
        
        self.canvas.draw()
    
    def select_all_values(self):
        """Select all value checkboxes"""
        for var in self.value_checkbox_vars:
            var.set(True)
        self.update_graph_visibility()
    
    def deselect_all_values(self):
        """Deselect all value checkboxes"""
        for var in self.value_checkbox_vars:
            var.set(False)
        self.update_graph_visibility()
    
    def toggle_all_values(self):
        """Toggle all value checkboxes"""
        for var in self.value_checkbox_vars:
            var.set(not var.get())
        self.update_graph_visibility()
    
    def apply_y_range(self):
        """Apply manual Y-axis range"""
        self.auto_scale_var.set(False)
        try:
            y_min = float(self.y_min_var.get())
            y_max = float(self.y_max_var.get())
            self.ax.set_ylim(y_min, y_max)
            self.canvas.draw()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for Y range.")
    
    def save_graph_image(self):
        """Save the current graph as an image"""
        default_filename = f"Graph_{time.strftime('%d%m_%H%M')}.png"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=default_filename
        )
        
        if not file_path:
            return
        
        try:
            self.fig.savefig(file_path)
            messagebox.showinfo("Success", f"Graph saved to {os.path.basename(file_path)}")
            self.status_var.set(f"Graph saved to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save graph: {str(e)}")
    
    def handle_disconnect_error(self, error_message):
        """Handle disconnection errors in the main thread"""
        messagebox.showerror("Connection Error", error_message)
        self.disconnect_from_port()
    
    def clear_data(self):
        """Clear all displayed data"""
        self.data_text.delete(1.0, tk.END)
        self.data_count = 0
        self.count_var.set("0")
        self.raw_data = []
        
        # Clear graph data
        for buffer in self.data_buffers:
            buffer.clear()
        self.time_buffer.clear()
        
        # Clear graph lines
        for line in self.lines:
            line.set_data([], [])
            line.set_visible(True)  # Reset visibility when clearing
        
        # Reset checkboxes to all selected
        for var in self.value_checkbox_vars:
            var.set(True)
        
        self.update_graph_visibility()
        self.canvas.draw()
    
    def save_data(self):
        """Save the displayed data to a file"""
        if not self.raw_data and self.data_text.get(1.0, tk.END).strip() == "":
            messagebox.showinfo("Info", "No data to save.")
            return
        
        default_filename = f"Data_{time.strftime('%d%m_%H%M')}.txt"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_filename
        )
        
        if not file_path:
            return
                
        try:
            if file_path.endswith('.csv'):
                # Save as CSV format
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write("Timestamp,Value1,Value2,Value3,Value4,Value5,Value6,Value7,Value8\n")
                    for timestamp, data in self.raw_data:
                        try:
                            values = [x.strip() for x in data.split(',')]
                            if len(values) == 8:
                                file.write(f"{timestamp},{','.join(values)}\n")
                        except:
                            pass
            else:
                # Save as text format
                with open(file_path, 'w', encoding='utf-8') as file:
                    for timestamp, data in self.raw_data:
                        file.write(f"[{timestamp}] {data}\n")
            
            messagebox.showinfo("Success", f"Data saved to {os.path.basename(file_path)}")
            self.status_var.set(f"Data saved to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")
    
    def on_closing(self):
        """Handle window closing event"""
        if self.is_connected:
            self.disconnect_from_port()
        plt.close('all')  # Close matplotlib figures
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ArduinoApp(root)
    root.mainloop()