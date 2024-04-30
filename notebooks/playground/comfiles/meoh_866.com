%nproc=4
%mem=5760MB
%chk=meoh_866.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4242 -0.0031 0.0053
C 0.0133 -0.0051 0.0073
H 1.7387 0.8982 -0.2164
H -0.3572 0.2517 -0.9851
H -0.3237 -0.9990 0.3018
H -0.3637 0.7446 0.7030

