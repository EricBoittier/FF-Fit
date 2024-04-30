%nproc=4
%mem=5760MB
%chk=meoh_993.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4065 0.0776 0.0356
C 0.0331 -0.0106 0.0192
H 1.9009 -0.2909 -0.7262
H -0.4748 0.7621 0.5964
H -0.3698 0.2182 -0.9674
H -0.3170 -1.0099 0.2779

