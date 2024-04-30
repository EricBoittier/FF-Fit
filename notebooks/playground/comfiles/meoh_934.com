%nproc=4
%mem=5760MB
%chk=meoh_934.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4417 0.1030 -0.0628
C -0.0005 -0.0239 0.0092
H 1.6053 -0.4737 0.7125
H -0.4241 0.9434 -0.2609
H -0.2970 -0.8522 -0.6343
H -0.3029 -0.1821 1.0444

