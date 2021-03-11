# FABLE
Fast rAdio Burst Localization &amp; dEtection using Mask-RCNN

UNDER DEVELOPMENT
---
Use the Jupyter notebook  to automate the process of masking for FRB or Single Pulse candidates

[![Masking Video](gif/masking.gif)](https://drive.google.com/file/d/11vHLk3XAHtV3lCx5VFZc9A4mro7KQFlp/view?usp=sharing)

Work under progress: Initial Results
---

[![Prediction Video](gif/prediction.gif)](https://drive.google.com/file/d/1Q_p0dpCvVITOJwRyDwiuhWo-OCST8SLf/view?usp=sharing)


Set up the Environment
---
- Create a Virtual environment using conda

         Create virtual environment: conda create -n frbloc python==3.6

- Activate the environment

        source activate frbloc or conda activate frbloc
       
- Install tensorflow and keras 

        pip3 install tensorflow-gpu==1.14
        pip3 install keras==2.1.0
        
- Install all required packages as follows

        pip3 install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug Ipython
        
- If jupyter can’t find the tensorflow, on terminal type

        python -m ipykernel install --user —name=frbloc,
        
- Then in Jupyter notebook choose frbloc kernel.
        

## Citing this work
___

If you use this work please cite:

    @software{zafiirah_hosenie_2020,
      author       = {Zafiirah Hosenie},
      title        = {{Zafiirah13/FABLE/: Software release}},
      month        = September,
      year         = 2020,
      publisher    = {Github},
      version      = {0.1},
      url          = {https://github.com/Zafiirah13/FABLE/}
    }
