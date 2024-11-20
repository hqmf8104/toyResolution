import multiprocessing
import time
import random

# THIS IS AN EXAMPLE OF MULTI PROCESSER PROCESSING

def image_processing(queue):
    """Simulates processing images and produces a list of dictionaries."""
    while True:
        image_data = [{"id": i, "value": f"Image-{random.randint(1, 100)}"} for i in range(3)]
        print(f"[Image Process] Captured: {image_data}")
        queue.put(("image", image_data))
        time.sleep(2)  # Simulate a delay

def chat_monitoring(queue):
    """Simulates pulling chat history and produces a list of dictionaries."""
    while True:
        chat_messages = [{"message_id": i, "text": f"Chat-{random.randint(1, 100)}"} for i in range(3)]
        print(f"[Chat Process] Fetched: {chat_messages}")
        queue.put(("chat", chat_messages))
        time.sleep(3)  # Simulate a delay

def output_processing(queue):
    """Combines outputs from the other two processes, ensuring no duplicates."""
    combined_data = []  # This will hold the combined list of dictionaries
    unique_entries = set()  # Set to track unique keys (e.g., "id" or "message_id")

    while True:
        if not queue.empty():
            source, data_list = queue.get()
            print(f"[Output Processor] Received {source} data: {data_list}")

            # Deduplication logic
            for item in data_list:
                # Assuming "id" for images and "message_id" for chat messages are unique keys
                unique_key = item.get("id") if "id" in item else item.get("message_id")
                
                if unique_key not in unique_entries:
                    combined_data.append(item)
                    unique_entries.add(unique_key)

            print(f"[Output Processor] Combined Data (no duplicates): {combined_data}")

if __name__ == "__main__":
    # Create a Queue for inter-process communication
    queue = multiprocessing.Queue()

    # Create and start the processes
    image_process = multiprocessing.Process(target=image_processing, args=(queue,))
    chat_process = multiprocessing.Process(target=chat_monitoring, args=(queue,))
    output_process = multiprocessing.Process(target=output_processing, args=(queue,))

    image_process.start()
    chat_process.start()
    output_process.start()

    # Optionally join processes (this example runs indefinitely)
    image_process.join()
    chat_process.join()
    output_process.join()
