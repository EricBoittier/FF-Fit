papermill ../templates/sim_template.ipynb ../out_notebooks/sims2_shake_water_k275.ipynb -k pycharmm -p jobpath  "/home/boittier/pcbach/sims2/shake/water/k275/dynamics.log" -p NSAVC 1000 -p dt 0.002

jupyter nbconvert --to webpdf --no-input ../out_sims2_shake_water_k275.ipynb
