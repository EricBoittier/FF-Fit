%nproc=4
%mem=5760MB
%chk=meoh_799.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4221 0.0945 -0.0497
C 0.0171 -0.0021 0.0067
H 1.8128 -0.5475 0.5792
H -0.3733 -0.4859 -0.8886
H -0.2823 -0.6189 0.8541
H -0.4749 0.9615 0.1394

