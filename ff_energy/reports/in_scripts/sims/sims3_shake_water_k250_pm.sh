papermill templates/sim_template.ipynb out_notebooks/sims/sims3_shake_water_k250.ipynb -k pycharmm -p jobpath  "/home/boittier/pcbach/sims3/shake/water/k250/dynamics.log" -p NSAVC 1000 -p dt 0.002

jupyter nbconvert --to webpdf --no-input out_notebooks/sims/sims3_shake_water_k250.ipynb
mv out_notebooks/sims/sims3_shake_water_k250.pdf out_pdfs/sims
mv sims3_shake_water_k250*csv out_csvs/sims
