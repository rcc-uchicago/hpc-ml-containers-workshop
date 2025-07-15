# 🧑‍💻 Combined Hands-On: Containerizing ML Workloads on HPC

> **Key Takeaways**
> - This unified guide walks you from basic container operations to advanced ML workflows on HPC.
> - All data is either generated within scripts or downloaded automatically using standard ML libraries—no manual downloads required.
> - Designed for both beginners and advanced users, with clear progression and troubleshooting support.

---

## Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [Part A: Core 60-Minute Lab](#part-a-core-60-minute-lab)
    - 0. Environment Sanity Check
    - 1. Basic Container Operations
    - 2. File Binding Patterns
    - 3. Interactive ML Training
    - 4. Batch (SLURM) Workflow
    - 5. Troubleshooting & Best Practices
3. [Part B: Advanced/Deep-Dive Lab](#part-b-advanceddeep-dive-lab)
    - 6. Real-World Dataset & CNN (CIFAR-10)
    - 7. Job Array Hyper-Parameter Sweep
    - 8. Distributed Training (Horovod/DDP)
    - 9. Security & Image Hardening
    - 10. CI/CD & Registry Push
4. [Reference & Best Practices](#reference--best-practices)
5. [Where to Get the Data](#where-to-get-the-data)
6. [Summary Table of Exercises](#summary-table-of-exercises)

---

## Prerequisites & Setup

- **Linux CLI basics**
- **Active cluster account**
- **Ability to load the Apptainer module** (`module load apptainer`)
- **GPU node access** for CUDA steps (CPU-only alternatives are always provided)
- **Recommended**: Access to a shared directory (e.g., `$SCRATCH`, `$HOME`, or `/output/`)

---

## Part A: Core 60-Minute Lab

| Time | Exercise | Goal |
|------|----------|------|
| 05 m | 0. Environment Sanity Check | Verify Apptainer, quota, GPU |
| 10 m | 1. Basic Apptainer Usage   | Pull / inspect / shell |
| 10 m | 2. File Binding Patterns   | Host ↔ container data flow |
| 15 m | 3. Interactive ML Training | PyTorch & TensorFlow, CPU+GPU |
| 15 m | 4. Batch (SLURM) Workflow  | Containerized jobs on the scheduler |
| 05 m | 5. Troubleshooting & Best Practices | Quick cheat-sheet |

---

### 0. Environment Sanity Check (5 min)

```bash
#!/usr/bin/env bash
module load apptainer || { echo "Apptainer module missing"; exit 1; }
echo -e "\n✓ Apptainer version: $(apptainer --version)"

# Optional GPU sanity (won't fail on CPU node)
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || echo "No GPUs visible (OK on CPU nodes)"
fi

# Disk quota check (warn if <5 GB free)
FREE=$(df -h $HOME | awk 'NR==2 {print $4}')
echo "Free space in $HOME: $FREE (need ≥5 GB)"
```

---

### 1. Basic Container Operations (10 min)

```bash
export IMAGES=$SCRATCH/$USER/sif        # or $HOME/sif if scratch unavailable
export WORKDIR=$SCRATCH/$USER/ml_work   # one stop for all scripts / logs
mkdir -p $IMAGES $WORKDIR && cd $WORKDIR

# Pull minimal images (or copy from /shared/containers/ if available)
apptainer pull $IMAGES/pytorch.sif    docker://pytorch/pytorch:2.2.2-cpu
apptainer pull $IMAGES/tf-gpu.sif     docker://tensorflow/tensorflow:latest-gpu

# Inspect container metadata
apptainer inspect $IMAGES/pytorch.sif

# Enter the PyTorch container's shell and test PyTorch
apptainer shell $IMAGES/pytorch.sif
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
exit
```

---

### 2. File Binding Patterns (10 min)

```bash
# Pattern 1: Bind a host directory to the same path inside the container
apptainer exec --bind $WORKDIR:$WORKDIR $IMAGES/pytorch.sif ls $WORKDIR

# Pattern 2: Bind a host directory to a different path inside the container
apptainer exec --bind $WORKDIR:/mnt $IMAGES/pytorch.sif ls /mnt

# Pattern 3: Bind multiple directories
apptainer exec --bind $WORKDIR:/work,$IMAGES:/images $IMAGES/pytorch.sif ls /work
```

---

### 3. Interactive ML Training (15 min)

#### PyTorch Example (Synthetic Data)

```bash
cat >polyfit.py <<'PY'
import torch
x = torch.linspace(-1, 1, 100).unsqueeze(1)
y = 3 * x ** 2 + 2 * x + 1 + 0.1 * torch.randn(x.size())
model = torch.nn.Sequential(torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(100): optimizer.zero_grad(); loss_fn(model(x), y).backward(); optimizer.step()
print("Done training.")
PY

apptainer exec --bind $WORKDIR:$WORKDIR $IMAGES/pytorch.sif python3 $WORKDIR/polyfit.py
```

#### TensorFlow Example (MNIST, Downloaded Automatically)

```bash
cat >mnist_tf.py <<'PY'
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train = x_train / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)
print("Done training.")
PY

apptainer exec --nv --bind $WORKDIR:$WORKDIR $IMAGES/tf-gpu.sif python3 $WORKDIR/mnist_tf.py
```

---

### 4. Batch (SLURM) Workflow (15 min)

#### Example: Single-GPU PyTorch Job

```bash
cat >run_pytorch.slurm <<'SLURM'
#!/bin/bash
#SBATCH --job-name=pytorch-job
#SBATCH --output=pt_out.txt
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

module load apptainer
apptainer exec --nv --bind $WORKDIR:$WORKDIR $IMAGES/pytorch.sif python3 $WORKDIR/polyfit.py
SLURM

sbatch run_pytorch.slurm
```

#### Example: Multi-GPU TensorFlow Job

```bash
cat >run_tf.slurm <<'SLURM'
#!/bin/bash
#SBATCH --job-name=tf-job
#SBATCH --output=tf_out.txt
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00

module load apptainer
apptainer exec --nv --bind $WORKDIR:$WORKDIR $IMAGES/tf-gpu.sif python3 $WORKDIR/mnist_tf.py
SLURM

sbatch run_tf.slurm
```

---

### 5. Troubleshooting & Best Practices (5 min)

- **Check GPU access:**  
  `apptainer exec --nv $IMAGES/pytorch.sif python3 -c "import torch; print(torch.cuda.is_available())"`
- **File system binding:**  
  Ensure your data/scripts are accessible inside the container.
- **Environment variables:**  
  Pass with `--env` or set inside the container.
- **Cache management:**  
  `apptainer cache list` and `apptainer cache clean`
- **Resource monitoring:**  
  Use `htop`, `nvidia-smi`, or `squeue` to monitor jobs.

---

## Part B: Advanced/Deep-Dive Lab

| Time | Exercise | Goal |
|------|----------|------|
| 10 m | 6. Real-World Dataset & CNN | CIFAR-10 download & training |
| 15 m | 7. Job Array Hyper-Parameter Sweep | Automate dozens of runs |
| 20 m | 8. Distributed Training (Horovod/DDP) | Multi-node, multi-GPU scaling |
| 10 m | 9. Security & Image Hardening | Modern container security on HPC |
| 05 m | 10. CI/CD & Registry Push | GitHub Actions→SIF deployment |

---

### 6. Real-World Dataset & CNN (CIFAR-10)

```bash
export IMAGES=$SCRATCH/$USER/sif
export WORKDIR=$SCRATCH/$USER/ml_work
mkdir -p $WORKDIR/cifar && cd $WORKDIR/cifar

# Download CIFAR-10 (only once, ~170 MB)
apptainer exec $IMAGES/pytorch.sif python3 -c "
import torchvision
_ = torchvision.datasets.CIFAR10(root='data', train=True, download=True)
print('Downloaded CIFAR-10 into ./data')
"

# Minimal CNN training script (cnn_cifar.py)
cat >cnn_cifar.py <<'PY'
import torch, torchvision, torchvision.transforms as T, torch.nn as nn
import torch.optim as optim
bs=128
trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=False,
        transform=T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.fc1 = nn.Linear(32*8*8, 64)
    self.fc2 = nn.Linear(64, 10)
  def forward(self, x):
    x = nn.functional.relu(self.conv1(x))
    x = nn.functional.max_pool2d(x, 2)
    x = nn.functional.relu(self.conv2(x))
    x = nn.functional.max_pool2d(x, 2)
    x = x.view(-1, 32*8*8)
    x = nn.functional.relu(self.fc1(x))
    return self.fc2(x)
net = Net()
optimizer = optim.Adam(net.parameters())
loss_fn = nn.CrossEntropyLoss()
for epoch in range(2):
  for i, (inputs, labels) in enumerate(trainloader):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
print('Done training.')
PY

apptainer exec --nv --bind $WORKDIR:$WORKDIR $IMAGES/pytorch.sif python3 $WORKDIR/cifar/cnn_cifar.py
```

---

### 7. Job Array Hyper-Parameter Sweep

Automate multiple runs with different learning rates using SLURM job arrays.

```bash
cat >sweep.slurm <<'SLURM'
#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --output=sweep_%A_%a.out
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

LRS=(0.1 0.01 0.001 0.0001 0.00001)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

module load apptainer
apptainer exec --nv --bind $WORKDIR:$WORKDIR $IMAGES/pytorch.sif \
  python3 $WORKDIR/cifar/cnn_cifar.py --lr $LR
SLURM

sbatch sweep.slurm
```

---

### 8. Distributed Training (Horovod/DDP)

Scale training across multiple nodes using Horovod or PyTorch DDP. (Requires multi-node allocation and appropriate container.)

- Example SLURM script and training code are provided in the advanced materials.
- Ensure your container includes Horovod or DDP support.

---

### 9. Security & Image Hardening

- Scan images for vulnerabilities (e.g., with `trivy`).
- Use minimal base images.
- Avoid running as root inside containers.
- Manage secrets securely (never hard-code passwords or tokens).

---

### 10. CI/CD & Registry Push

- Use GitHub Actions or similar CI/CD tools to automate building and pushing `.sif` images to a registry.
- Example workflows are available in the reference appendix.

---

## Reference & Best Practices

- **Use versioned container images** (avoid `latest` for reproducibility).
- **Store code and data outside containers** for persistence.
- **Automate environment checks** at the start of every session.
- **Document data acquisition** clearly in scripts.
- **Support both CPU and GPU workflows** with clear instructions.
- **Provide ready-to-use SLURM scripts** for batch jobs.
- **Include troubleshooting and best practices** at the end of your materials.

---

## Where to Get the Data

| Exercise Section                | Data Source/Handling Method                                                                 |
|----------------------------------|-------------------------------------------------------------------------------------------|
| Core ML Training (PyTorch, TF)   | Synthetic data generated within scripts                                                   |
| TensorFlow MNIST Example         | MNIST loaded via TensorFlow's built-in loader                                             |
| Advanced CNN (CIFAR-10)          | CIFAR-10 downloaded using `torchvision.datasets.CIFAR10` in script                       |
| Container Images                 | Pulled from Docker Hub or copied from `/shared/containers/` if available                  |

> **No manual data download is required.**  
> All datasets are either generated on the fly or downloaded automatically by the provided scripts using standard ML library APIs.

---

## Summary Table of Exercises

| Part | Audience | Duration | Description |
|------|----------|----------|-------------|
| A | Everyone | 60 min | Core containerization lab (Exercises 0–5) |
| B | Power users | 60–90 min | Advanced topics (Exercises 6–10) |
| C | Reference | — | Additional examples, monitoring, best practices |

---

> **Key Takeaway:**  
> This combined hands-on guide is self-contained and ready for use on any HPC system with Apptainer. All data is handled automatically, and the exercises progress from basic container usage to advanced ML workflows, ensuring a smooth learning curve for all users.

---

**For more advanced topics, troubleshooting, and up-to-date scripts, visit the official workshop GitHub repository or consult your cluster's documentation.**
