%nproc=4
%mem=5760MB
%chk=meoh_395.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4330 -0.0045 0.0116
C -0.0123 0.0085 0.0107
H 1.7862 0.8300 -0.3617
H -0.3754 1.0162 0.2126
H -0.2795 -0.3778 -0.9729
H -0.2722 -0.7130 0.7853

