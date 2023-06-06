papermill templates/sim_template.ipynb out_notebooks/sims/sims_optpc_water_k275.ipynb -k pycharmm -p jobpath  "/home/boittier/pcbach/sims/optpc/water/k275/dynamics.log" -p NSAVC 1000 -p dt 0.0002

jupyter nbconvert --to webpdf --no-input out_notebooks/sims/sims_optpc_water_k275.ipynb
mv out_notebooks/sims/sims_optpc_water_k275.pdf out_pdfs/sims
mv sims_optpc_water_k275*csv out_csvs/sims
