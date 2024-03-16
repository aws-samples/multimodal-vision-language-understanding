yum install unzip -y
conda install -y conda
conda install -c conda-forge git-lfs -y
git lfs install

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
git clone git@hf.co:datasets/liuhaotian/LLaVA-Instruct-150K