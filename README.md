# ProtoVerse: ~

**ProtoVerse** is a unified simulation and learning framework for physics-based humanoid animation and control.  
It merges capabilities from two powerful libraries:
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions) – for interactive character animation and learning.
- [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse) – for humanoid simulation and multi-task learning.

---


# Install Instructions (Isaaclab)
```
git clone --recurse-submodules https://github.com/osheraz/ProtoVerse.git

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