papermill ../templates/sim_template.ipynb ../out_notebooks/sims4_optpc_water_k225.ipynb -k pycharmm -p jobpath  "/home/boittier/pcbach/sims4/optpc/water/k225/dynamics.log" -p NSAVC 1000 -p dt 0.0002

jupyter nbconvert --to webpdf --no-input ../out_sims4_optpc_water_k225.ipynb
