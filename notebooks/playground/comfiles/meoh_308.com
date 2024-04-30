%nproc=4
%mem=5760MB
%chk=meoh_308.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4447 0.0037 0.0480
C 0.0037 0.0001 -0.0074
H 1.5551 0.6798 -0.6529
H -0.3236 0.2603 -1.0140
H -0.3696 -0.9778 0.2965
H -0.3789 0.7621 0.6717

