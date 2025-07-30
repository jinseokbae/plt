# install human_body_prior to use it while retargeting
cd ../
git clone https://github.com/nghorbani/human_body_prior.git
cd human_body_prior
pip install -e .
cd ../hybrid_latent_motion_prior
# reinstall torch
pip uninstall torch
pip uninstall torchvision
pip uninstall torchaudio
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# install torch scatter
pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html 
pip install -e .