papermill templates/sim_template.ipynb out_notebooks/sims/sims4_shake_water_k275.ipynb -k pycharmm -p jobpath  "/home/boittier/pcbach/sims4/shake/water/k275/dynamics.log" -p NSAVC 1000 -p dt 0.002

jupyter nbconvert --to webpdf --no-input out_notebooks/sims/sims4_shake_water_k275.ipynb
mv out_notebooks/sims/sims4_shake_water_k275.pdf out_pdfs/sims
mv sims4_shake_water_k275*csv out_csvs/sims
