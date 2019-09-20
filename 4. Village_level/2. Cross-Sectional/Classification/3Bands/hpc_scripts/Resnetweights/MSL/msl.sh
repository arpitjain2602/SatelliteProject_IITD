#PBS -P cse
#PBS -q low
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=120:00:00
module load "apps/pythonpackages/2.7.13/tensorflow/1.5.0/gpu"
module load "pythonpackages/2.7.13/ucs4/gnu/447/keras/2.0.3/gnu"
module load "pythonpackages/2.7.13/ucs4/gnu/447/pandas/0.20.0rc1/gnu"
module load "pythonpackages/2.7.13/ucs4/gnu/447/h5py/2.7.0/gnu"
pwd
cd /home/cse/mtech/mcs172873
python /home/cse/mtech/mcs172873/Inceptionnet_V3_Deep_Learning/scripts_arpit/iitdproxy.py proxy &
#python iitdproxy.py proxy &
export http_proxy=10.10.78.62:3128
export https_proxy=10.10.78.62:3128
pip install --upgrade pip
pip install tensorflow
#pip install gedit
#pip install --upgrade pip
#pip install libtiff
#git config --global http.proxy http://proxy22.iitd.ernet.in:3128
#git config --global https.proxy https://proxy22.iitd.ernet.in:3128
#pip install --user --upgrade tqdm
#pip install --user h5py
pwd
python /home/cse/mtech/mcs172873/LANDSAT/Resnetweights/MSL/msl.py
#git clone https://github.com/fizyr/keras-retinanet.git
#git push
#pip install --user boltons
#git clone https://github.com/LuaDist/qtlua
#git clone https://github.com/torch/distro.git ~/torch --recursive
#cd ./torch
#bash install-deps
#./install.sh
#git clone https://github.com/BVLC/caffe
#pip install --user --upgrade libtiff