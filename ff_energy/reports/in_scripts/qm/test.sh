papermill templates/qm_template.ipynb out_notebooks/qm/test.ipynb -k pycharmm -p dataname water_cluster/pbe0dz/pbe0_dz.pc

jupyter nbconvert --to webpdf --no-input out_notebooks/qm/test.ipynb
mv out_notebooks/qm/test.pdf out_pdfs/qm
