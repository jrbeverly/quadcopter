conda install jupyter keras tensorflow -y --quiet 
pip install tensorflow
cd /opt/notebooks 
jupyter notebook --allow-root --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser