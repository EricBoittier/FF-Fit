%nproc=4
%mem=5760MB
%chk=meoh_143.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4221 0.0222 0.0431
C 0.0196 -0.0093 0.0100
H 1.7743 0.5260 -0.7202
H -0.3836 -0.9704 -0.3093
H -0.4464 0.2819 0.9513
H -0.2922 0.7060 -0.7510

