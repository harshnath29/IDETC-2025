import serial
import time

class HotbedController:
    def __init__(self, port='/dev/ttyACM0', baud_rate=9600):
        self.ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Allow time for connection to establish

    def read_data(self):
        if self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8').strip()
            return self.parse_data(line)
        return None

    def parse_data(self, data):
        parts = data.split()
        if len(parts) == 6:
            return {
                'ADC': int(parts[1]),
                'Temp': float(parts[3]),
                'Heater': parts[5]
            }
        return None

    def set_temperature(self, temp):
        command = f"SET_TEMP {temp}\n"
        self.ser.write(command.encode())

    def close(self):
        self.ser.close()
