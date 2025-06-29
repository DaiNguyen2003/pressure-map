import serial
import time
import serial.tools.list_ports

def list_available_ports():
    ports = serial.tools.list_ports.comports()
    print("Available ports:")
    for port in ports:
        print(f"- {port.device}")
    return [port.device for port in ports]

available_ports = list_available_ports()

if not available_ports:
    print("No COM ports found. Please check your connections.")
    exit()

print("Using port:", available_ports[0])  # Default to first available port
port_to_use = available_ports[0]  # You can change this if needed

try:
    # Connect to Arduino's serial port
    ser = serial.Serial(port_to_use, 9600, timeout=1)
    print("Connected to Arduino...")
    time.sleep(3)  # Wait for Arduino to initialize
    
    # Read and display data from Arduino
    print("Reading data from HX711 load cell:")
    print("--------------------------------")
    
    while True:
        try:
            data = ser.readline().decode('utf-8').strip()
            if data:
                print(data)
        except KeyboardInterrupt:
            print("\nProgram stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
            
except serial.SerialException as e:
    print(f"Failed to connect to {port_to_use}: {e}")
    print("Please check if the Arduino is properly connected and the correct port is selected.")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial connection closed")