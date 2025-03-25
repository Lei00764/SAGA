# 🤖 SAGA: Semi-Autonomous Arm-Hand Teleoperation with Grasping Assistance

This is the official code for the paper "Semi-Autonomous Arm-Hand Teleoperation with Grasping Assistance", which introduces a two-stage teleoperation framework for increasing operational efficiency.

### 🛠️ Installation

1. Download and install [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) from NVIDIA's website

2. Verify Isaac Gym installation:

```bash
cd isaac-gym/python/examples
python joint_monkey.py
```

3. Clone and install this repository:

```bash
git clone https://github.com/lei00764/SAGA
cd SAGA
pip install -r requirements.txt
```

## 🚀 Running

### Training

```bash
python DexHandEnv/train.py task=DexCube num_envs=4096 headless=True
```
- `num_envs`: Number of parallel environments (default: 4096)
- `headless`: Run without visualization for faster training

### Testing

To test a trained model:

```bash
python DexHandEnv/train.py task=DexCube test=True num_envs=1 checkpoint=$(find $(ls -td runs/DexCube_* | head -n 1) -name "DexCube.pth")
```

### Configuration

The environment and training parameters can be customized through config files:

- Environment config: `DexHandEnv/config/task/DexCube.yaml`
- Training config: `DexHandEnv/config/train/DexCubePPO.yaml`

### Video Recording

To capture training videos:

```bash
python DexHandEnv/train.py task=DexCube capture_video=True capture_video_freq=1500 capture_video_len=100
```

### Multi-GPU Training

For distributed training across multiple GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 DexHandEnv/train.py multi_gpu=True task=DexCube
```

## 🥰 Acknowledgements

This work builds upon the Isaac Gym framework developed by NVIDIA.

## Contact
If you have any questions, please contact Xiang Lei at [2053932@tongji.edu.cn](2053932@tongji.edu.cn).
