import torch
import faiss


def print_gpu_status():
	print("CUDA available:", torch.cuda.is_available())
	print("CUDA device count:", torch.cuda.device_count())

	if torch.cuda.is_available():
		device_id = torch.cuda.current_device()
		print("Current device:", device_id)
		print("Device name:", torch.cuda.get_device_name(device_id))
		print("Memory allocated (MB):", round(torch.cuda.memory_allocated(device_id) / 1024**2, 2))
		print("Memory reserved (MB):", round(torch.cuda.memory_reserved(device_id) / 1024**2, 2))

	print("FAISS GPU count:", faiss.get_num_gpus())


if __name__ == "__main__":
	print_gpu_status()