papermill templates/sim_template.ipynb out_notebooks/sims/sims4_kmdcm_water_k289.ipynb -k pycharmm -p jobpath  "/home/boittier/pcbach/sims4/kmdcm/water/k289/dynamics.log" -p NSAVC 1000 -p dt 0.0002

jupyter nbconvert --to webpdf --no-input out_notebooks/sims/sims4_kmdcm_water_k289.ipynb
mv out_notebooks/sims/sims4_kmdcm_water_k289.pdf out_pdfs/sims
mv sims4_kmdcm_water_k289*csv out_csvs/sims
