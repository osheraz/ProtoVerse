# ProtoVerse: ~

**ProtoVerse**   
It merges capabilities from two libraries:
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions) – for interactive character animation and learning.
- [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse) – for humanoid simulation and multi-task learning.

---


# Install Instructions 


```
git clone --recurse-submodules https://github.com/osheraz/ProtoVerse.git

git lfs fetch --all  # to fetch all files stored in git-lfs.

```

## (Isaaclab)

```
conda create -n proto_sim python==3.10
conda activate proto_sim
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

cd ProtoVerse
pip install -e .

cd dependencies/IsaacLab
./isaaclab.sh --install none

```

## Isaacgym

mamba env create -n proto_gym python==3.8
pip install -e isaacgym/python
pip install -e isaac_utils
pip install -e poselib

cd ProtoVerse
pip install -e .
pip install -r requirements_isaacgym.txt
