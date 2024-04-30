%nproc=4
%mem=5760MB
%chk=meoh_158.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4298 0.0355 0.0644
C 0.0185 0.0065 -0.0155
H 1.6938 0.2193 -0.8614
H -0.3161 -0.9796 -0.3378
H -0.3911 0.1403 0.9857
H -0.4429 0.7645 -0.6484

