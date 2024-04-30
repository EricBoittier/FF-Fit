%nproc=4
%mem=5760MB
%chk=meoh_706.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4250 0.0367 0.0467
C 0.0269 0.0012 0.0130
H 1.6977 0.2667 -0.8662
H -0.2858 -0.9201 -0.4786
H -0.4854 0.0138 0.9750
H -0.4058 0.8277 -0.5507

