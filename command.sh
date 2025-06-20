#!/bin/bash
pip install "cython<3"

conda install -y -c conda-forge xorg-libx11

conda install -y -c conda-forge mesalib

conda install -y -c conda-forge glew

pip install patchelf

pip install PyYAML
 
# install mojuco
pip install gym[mujoco]

pip install -U 'mujoco-py<2.2,>=2.1'

pip install mujoco==2.3.3