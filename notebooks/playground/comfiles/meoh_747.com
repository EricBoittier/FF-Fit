%nproc=4
%mem=5760MB
%chk=meoh_747.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4218 0.0949 0.0292
C 0.0132 -0.0072 0.0148
H 1.8056 -0.5415 -0.6097
H -0.2772 -0.7611 -0.7169
H -0.3813 -0.2659 0.9974
H -0.4145 0.9324 -0.3351

