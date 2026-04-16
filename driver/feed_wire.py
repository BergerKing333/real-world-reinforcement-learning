import json
import math
import serial
import time


SERIAL_PORT = '/dev/ttyUSB0'
SERIAL_BAUD = 115200

SERIAL_TIMEOUT   = 0.1    # s — readline timeout (short so we can check deadline)
MOTION_TIMEOUT_S = 30.0   # s — max time to wait for POSITION_REACHED before giving up
SILENCE_TIMEOUT  = 3.0    # s — if device goes quiet after acknowledging, assume done


driver = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)

insert_amount_cm = -15

cmd_str = f"{(insert_amount_cm * 500):.2f}, 0.0"

driver.reset_input_buffer()

driver.write(cmd_str.encode('utf-8'))


deadline     = time.time() + MOTION_TIMEOUT_S
confirmed    = False
acknowledged = False
last_rx      = time.time()
# while time.time() < deadline:
#     raw = driver.readline()
#     if not raw:
#         # No data — check silence timeout after acknowledgment
#         if acknowledged and (time.time() - last_rx) >= SILENCE_TIMEOUT:
#             self.get_logger().info(
#                 f'Device silent for {SILENCE_TIMEOUT:.0f} s '
#                 'after ack — assuming motion complete.'
#             )
#             confirmed = True
#             break
#         continue
#     line = raw.decode('utf-8', errors='replace').strip()
#     if line:
#         last_rx = time.time()
#         print(f'← device: {line}')
#     if 'POSITION_REACHED' in line:
#         confirmed = True
#         break
#     if 'Parsed' in line or 'Received' in line:
#         acknowledged = True
