papermill ../templates/sim_template.ipynb ../out_notebooks/sims3_shake_water_k225.ipynb -k pycharmm -p jobpath  "/home/boittier/pcbach/sims3/shake/water/k225/dynamics.log" -p NSAVC 1000 -p dt 0.002

jupyter nbconvert --to webpdf --no-input ../out_sims3_shake_water_k225.ipynb
