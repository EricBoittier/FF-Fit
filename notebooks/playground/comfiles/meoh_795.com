%nproc=4
%mem=5760MB
%chk=meoh_795.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4188 0.1011 -0.0382
C 0.0276 -0.0069 -0.0030
H 1.8171 -0.6200 0.4927
H -0.4605 -0.4886 -0.8503
H -0.2605 -0.5785 0.8793
H -0.4861 0.9495 0.0944

