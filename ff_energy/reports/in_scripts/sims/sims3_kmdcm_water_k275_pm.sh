papermill ../templates/sim_template.ipynb ../out_notebooks/sims3_kmdcm_water_k275.ipynb -k pycharmm -p jobpath  "/home/boittier/pcbach/sims3/kmdcm/water/k275/dynamics.log" -p NSAVC 1000 -p dt 0.0002

jupyter nbconvert --to webpdf --no-input ../out_sims3_kmdcm_water_k275.ipynb
