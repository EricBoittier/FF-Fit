papermill templates/sim_template.ipynb out_notebooks/sims/sims_optpc_water_k300.ipynb -k pycharmm -p jobpath  "/home/boittier/pcbach/sims/optpc/water/k300/dynamics.log" -p NSAVC 1000 -p dt 0.0002

jupyter nbconvert --to webpdf --no-input out_notebooks/sims/sims_optpc_water_k300.ipynb
mv out_notebooks/sims/sims_optpc_water_k300.pdf out_pdfs/sims
mv sims_optpc_water_k300*csv out_csvs/sims
