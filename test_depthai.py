import depthai as dai
import time

max_retries = 3
retry_delay = 2  # seconds

for attempt in range(max_retries):
    try:
        with dai.Device() as device:
            print(f"Connected to device: {device.getDeviceName()}")
            # Your main code here
            break  # Exit the loop if connection is successful
    except dai.XLinkError as e:
        print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("Failed to connect to the device after multiple attempts.")
            raise