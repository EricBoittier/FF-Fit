%nproc=4
%mem=5760MB
%chk=meoh_524.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4315 0.0915 0.0385
C -0.0172 -0.0195 0.0044
H 1.8336 -0.4046 -0.7050
H -0.2997 -0.6565 0.8426
H -0.3018 1.0217 0.1559
H -0.2899 -0.3956 -0.9816

