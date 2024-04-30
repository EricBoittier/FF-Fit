%nproc=4
%mem=5760MB
%chk=meoh_396.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4278 -0.0018 0.0150
C -0.0037 0.0109 0.0117
H 1.8078 0.7906 -0.4189
H -0.4022 1.0015 0.2313
H -0.2792 -0.3747 -0.9699
H -0.2867 -0.7346 0.7548

