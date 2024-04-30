%nproc=4
%mem=5760MB
%chk=meoh_302.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4420 -0.0051 0.0324
C -0.0149 0.0055 -0.0111
H 1.6923 0.8288 -0.4175
H -0.3326 0.1389 -1.0452
H -0.2798 -0.9704 0.3958
H -0.3319 0.8043 0.6593

