%nproc=4
%mem=5760MB
%chk=meoh_761.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4398 0.1153 0.0196
C -0.0109 -0.0158 -0.0061
H 1.7381 -0.7444 -0.3444
H -0.3283 -0.7452 -0.7512
H -0.2599 -0.3504 1.0009
H -0.4144 0.9835 -0.1693

