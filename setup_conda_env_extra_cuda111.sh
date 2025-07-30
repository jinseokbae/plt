# install human_body_prior to use it while retargeting
cd ../
git clone https://github.com/nghorbani/human_body_prior.git
cd human_body_prior
pip install -e .
cd ../hybrid_latent_motion_prior
# install torch scatter
pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html 
pip install -e .