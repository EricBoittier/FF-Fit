papermill templates/ff_template.ipynb out_notebooks/ff/epol_mike_pbe0dz_FOE.ff.pkl.ipynb -k pycharmm -p ffpkl epol_mike_pbe0dz_FOE.ff.pkl

jupyter nbconvert --to webpdf --no-input out_notebooks/ff/epol_mike_pbe0dz_FOE.ff.pkl.ipynb
mv out_notebooks/ff/epol_mike_pbe0dz_FOE.ff.pkl.pdf out_pdfs/ff
