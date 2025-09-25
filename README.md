# ProtoVerse: ~
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
<img src="https://img.shields.io/github/last-commit/osheraz/ProtoVerse?style&color=5D6D7E" alt="git-last-commit" />

**ProtoVerse**   
It merges capabilities from two libraries:
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions) – for interactive character animation and learning.
- [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse) – for humanoid simulation and multi-task learning.

---

</div>
<br><br>
<div align="center">
  <img src="assets/feet.gif"
  width="80%">
</div>
<br>

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
```
mamba env create -n proto_gym python==3.8
pip install -e isaacgym/python
pip install -e isaac_utils
pip install -e poselib

cd ProtoVerse
pip install -e .
pip install -r requirements_isaacgym.txt
```