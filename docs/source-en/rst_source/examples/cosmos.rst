RL with Cosmos Predict 2.5 World Model
=======================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This guide walks through setting up **Cosmos Predict 2.5** as an action-conditioned
world model environment in RLinf on a **fresh machine**.
All required model artifacts are hosted on HuggingFace at
|huggingface| `tayalmanan/cosmos-robotics <https://huggingface.co/tayalmanan/cosmos-robotics>`_.

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
--------

Cosmos Predict 2.5 is an action-conditioned video generation model that can serve as
a learned simulator for robotic manipulation tasks. Given an initial RGB observation
and a sequence of actions, it generates future video frames that predict how the
environment will evolve.

In RLinf, Cosmos replaces the traditional physics simulator in the RL training loop:

1. The **VLA policy** proposes actions given the current observation.
2. **Cosmos** generates future frames conditioned on those actions (world model step).
3. A **reward model** scores the generated frames (success/fail).
4. The policy is updated via **GRPO** using the imagined trajectories.

Architecture summary:

- **DiT backbone**: 2B parameters (2048 hidden dim, 28 blocks, 16 heads)
- **Resolution**: 256×320 @ 4 FPS
- **Frame prediction**: 12 future frames per step (``chunk_size: 12``)
- **Denoising**: 10 steps with RectifiedFlow 2AB solver, CFG guidance = 7
- **Tokenizer**: Cosmos video VAE (encoder/decoder)


Prerequisites
-------------

- **GPU**: 4× A100 80GB (or equivalent; ~50GB VRAM per GPU during training)
- **CUDA**: 12.x
- **Python**: 3.10 (required by Cosmos Predict 2)
- **OS**: Linux (Ubuntu 20.04+)


Step 1 — Clone the repository
------------------------------

.. code-block:: bash

   git clone https://github.com/tayalmanan28/wm_vla_rl.git
   cd wm_vla_rl


Step 2 — Install the Cosmos virtual environment
-------------------------------------------------

RLinf uses separate virtual environments per component.
The installer will create ``.venv_cosmos`` with all Cosmos + LIBERO dependencies:

.. code-block:: bash

   bash requirements/install.sh cosmos_world_model

This installs:

- ``cosmos-predict2`` (from source, with ``--no-deps`` to avoid hydra conflicts)
- NVIDIA runtime wheels (megatron-core, transformer-engine, natten, etc.)
- flash-attn (< 2.8 for torch 2.6 ABI compat)
- LIBERO environment and robosuite
- The ``rlinf`` package itself

The VLA policy environment (e.g. OpenVLA-OFT) is installed separately:

.. code-block:: bash

   bash requirements/install.sh openvla_oft


Step 3 — Download model checkpoints
-------------------------------------

Download the self-contained Cosmos model directory from HuggingFace:

.. code-block:: bash

   pip install huggingface_hub

   python -c "
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id='tayalmanan/cosmos-robotics',
       local_dir='models/Cosmos-Predict2.5-LIBERO-Spatial',
   )
   "

This downloads (~12.5 GB total):

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - File
     - Size
     - Description
   * - ``libero-spatial-2b-19k.pt``
     - 11.89 GB
     - Cosmos 2B DiT checkpoint (19k training iterations)
   * - ``resnet_rm.pth``
     - 43 MB
     - ResNet binary reward model
   * - ``tokenizer/tokenizer.pth``
     - 485 MB
     - Cosmos video VAE tokenizer
   * - ``dataset/`` (400 ``.npy`` files)
     - 77 MB
     - LIBERO-Spatial initial-state trajectories for env reset
   * - ``dataset_statistics.json``
     - 2 KB
     - Action normalization statistics (mean / std)

You also need the VLA policy checkpoint:

.. code-block:: bash

   python -c "
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id='RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora',
       local_dir='models/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora',
   )
   "

After downloading, your ``models/`` directory should look like::

   models/
   ├── Cosmos-Predict2.5-LIBERO-Spatial/
   │   ├── libero-spatial-2b-19k.pt
   │   ├── resnet_rm.pth
   │   ├── tokenizer/
   │   │   └── tokenizer.pth
   │   ├── dataset/
   │   │   ├── seed_0_traj_0.npy
   │   │   ├── seed_0_traj_1.npy
   │   │   └── ... (400 files)
   │   └── dataset_statistics.json
   └── RLinf-OpenVLAOFT-LIBERO-130-Base-Lora/
       ├── model-00001.safetensors
       └── ...


Step 4 — Configure paths
--------------------------

The training config uses a single ``cosmos_model_dir`` variable. All sub-paths
(checkpoint, reward model, tokenizer, dataset) are resolved relative to it via
Hydra interpolation.

Open ``examples/embodiment/config/cosmos_libero_spatial_grpo_openvlaoft.yaml``
and verify the model paths:

.. code-block:: yaml

   env:
     train:
       cosmos_model_dir: "models/Cosmos-Predict2.5-LIBERO-Spatial"

   rollout:
     model:
       model_path: "models/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora"

   actor:
     model:
       model_path: "models/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora"

These are **relative to the RLinf repository root** (the working directory when
``run_embodiment.sh`` is executed).

The environment YAML (``config/env/cosmos_libero_spatial.yaml``) derives all
sub-paths automatically:

.. code-block:: yaml

   cosmos_model_dir: null   # overridden by top-level config

   initial_image_path: ${env.train.cosmos_model_dir}/dataset/

   world_model_cfg:
     stats_path: ${env.train.cosmos_model_dir}/dataset_statistics.json
     reward_model:
       type: ResnetRM
       from_pretrained: ${env.train.cosmos_model_dir}/resnet_rm.pth
     cosmos:
       ckpt_path: ${env.train.cosmos_model_dir}/libero-spatial-2b-19k.pt
       tokenizer_path: ${env.train.cosmos_model_dir}/tokenizer/tokenizer.pth


Step 5 — Set up Ray cluster
-----------------------------

RLinf uses Ray for distributed scheduling. On a single node:

.. code-block:: bash

   ray start --head --num-gpus=4 --num-cpus=32

For multi-node, see ``ray_utils/start_ray.sh``.


Step 6 — Launch training
--------------------------

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh cosmos_libero_spatial_grpo_openvlaoft

Expected output (first epoch on 4× A100 80GB):

.. code-block:: text

   Generating Rollout Epochs:   6%|▋  | 1/16 [08:56<2:14:09, 536.61s/it]

Each epoch takes ~9 minutes. A full rollout generation phase (16 epochs) takes
~2.5 hours.


Architecture Deep Dive
-----------------------

Code structure
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File
     - Purpose
   * - ``rlinf/envs/world_model/world_model_cosmos_env.py``
     - Main Cosmos WM environment class
   * - ``rlinf/envs/world_model/world_model_cosmos_offload_env.py``
     - Offloaded variant (DiT ↔ policy GPU time-sharing)
   * - ``rlinf/models/reward_model/resnet_rm.py``
     - Standalone ResNet reward model (no OpenSora dependency)
   * - ``examples/embodiment/config/env/cosmos_libero_spatial.yaml``
     - Environment-level configuration
   * - ``examples/embodiment/config/cosmos_libero_spatial_grpo_openvlaoft.yaml``
     - Full training config (algorithm + env + model)

Inference pipeline
~~~~~~~~~~~~~~~~~~~

Each world model step:

1. **Encode** current observation via Cosmos VAE tokenizer → latent
2. **Denoise** for 10 steps using 2AB solver (2 forward passes per step: conditional + unconditional for CFG) → 22 DiT forward passes total
3. **Decode** latent → predicted RGB frames via VAE decoder
4. **Score** final frame with ResNet reward model → binary {0, 1}

Batched inference
~~~~~~~~~~~~~~~~~~

All environments on a single GPU worker are batched together (``batch_size: 16``),
running through the DiT in a single forward pass per denoising step.
This reduces per-epoch time from ~55 min (sequential) to ~9 min.


Key configuration parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``cosmos.num_steps``
     - 10
     - Number of denoising steps (quality vs speed tradeoff)
   * - ``cosmos.guidance``
     - 7
     - Classifier-free guidance scale
   * - ``cosmos.action_scaler``
     - 20.0
     - Multiplier for normalized actions before DiT input
   * - ``cosmos.chunk_size``
     - 12
     - Future frames per world model step
   * - ``cosmos.batch_size``
     - 16
     - Micro-batch size for batched DiT inference
   * - ``condition_frame_length``
     - 4
     - Number of conditioning frames (context window)
   * - ``image_size``
     - [256, 320]
     - Output resolution (H × W)


Comparison with OpenSora
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * - Aspect
     - OpenSora
     - Cosmos Predict 2.5
   * - DiT size
     - 1.1B (STDiT3)
     - 2B
   * - Solver
     - 1st-order Euler
     - 2nd-order 2AB (RectifiedFlow)
   * - CFG
     - None (single pass)
     - Yes (2 passes per step)
   * - Steps
     - 10
     - 10
   * - DiT forward passes
     - 10
     - 22
   * - Epoch time (4×A100)
     - ~8 min
     - ~9 min
   * - Dependencies
     - OpenSora package
     - cosmos-predict2 only


Troubleshooting
----------------

**OOM errors**

- Ensure ``enable_offload: True`` is set in both ``env`` and ``rollout`` sections.
- Kill stale GPU processes: ``kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)``
- Check GPU memory: ~50GB per GPU is expected.

**Tokenizer not found**

- Verify ``tokenizer_path`` points to the correct file inside the model directory.
- The tokenizer is loaded via ``config.tokenizer.vae_pth`` override in
  ``world_model_cosmos_env.py``.

**Slow first epoch**

- The first epoch includes model loading time. Subsequent epochs should be ~9 min.
- Ensure flash-attn is installed: ``python -c "import flash_attn; print(flash_attn.__version__)"``

**Ray connection issues**

- Check Ray is running: ``ray status``
- Restart: ``ray stop --force && ray start --head --num-gpus=4``
