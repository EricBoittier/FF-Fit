papermill templates/ff_template.ipynb out_notebooks/ff/DUMMY_pbe0dz_clusters.ff.pkl.ipynb -k pycharmm -p ffpkl DUMMY_pbe0dz_clusters.ff.pkl

jupyter nbconvert --to webpdf --no-input out_notebooks/ff/DUMMY_pbe0dz_clusters.ff.pkl.ipynb
mv out_notebooks/ff/DUMMY_pbe0dz_clusters.ff.pkl.pdf out_pdfs/ff
