import torch


def check_cuda():
    """Check if CUDA is available and return the appropriate device(s)."""
    print(f"{' 🔎 Checking CUDA availability ':-^80}")

    print("     PyTorch version: ", torch.__version__)
    print("     CUDA Version: ", torch.version.cuda)

    if torch.cuda.is_available():
        print("     CUDA is available ✔️")
    else:
        print("     CUDA is not available ❌")

    print("     Available GPU devices: ", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f"      -Device {i}: {torch.cuda.get_device_name(i)}")

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("     Multiple GPUs detected!")
            print(80 * "-")
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        else:
            print("     Single GPU detected. Using the GPU.")
            print(80 * "-")
            return torch.device("cuda:0")
    else:
        print("     No GPU available. Using CPU.")
        print(80 * "-")
        return torch.device("cpu")