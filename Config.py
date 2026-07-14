import json

class Config:


    # DINO V3 Dataset Transformations
    # Crop sizes (height, width)
    global_size = (448, 160)
    local_size = (224, 128)

    # CHANGE 1: Allow smaller crops so the window can slide
    global_crops_scale = (0.4, 1.0)
    local_crops_scale = (0.05, 0.3)

    num_local_crops = 6

    # Aspect ratio margins applied around the target ratio
    ratio_margin_low = 0.9
    ratio_margin_high = 1.1

    # Color jitter values
    jitter_brightness = 0.4
    jitter_contrast = 0.4
    jitter_saturation = 0.2
    jitter_hue = 0.1

    # Augmentation probabilities
    horizontal_flip_p = 0.5
    color_jitter_p = 0.8
    grayscale_p = 0.2
    global_blur_1_p = 1.0
    global_blur_2_p = 0.1
    local_blur_p = 0.5

    # Gaussian blur parameters
    blur_kernel_size = 9
    blur_sigma = (0.1, 2.0)

    # Solarize parameters
    solarize_threshold = 128
    solarize_p = 0.2

    # Normalization statistics
    mean = [0.4114, 0.4183, 0.4359]
    std = [0.3005, 0.2981, 0.2966]



    # Model Config
    base_model = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    output_dim = 65536

    # Training Loop Config
    epochs = 100
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 0.04
    accumulation_steps = 32

    # Checkpointing
    checkpoint_dir = "./checkpoints"
    save_every_epochs = 1

    # Loss Weights
    lambda_dino = 1.0
    lambda_ibot = 1.0
    lambda_koleo = 0.1

    @staticmethod
    def get_config():
        """Returns the configuration as a dictionary."""
        return {
            key: value for key, value in Config.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

    @staticmethod
    def save_config(output_path: str):
        """Saves the configuration to a JSON file."""
        config = Config.get_config()
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)