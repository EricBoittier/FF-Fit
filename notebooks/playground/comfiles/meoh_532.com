%nproc=4
%mem=5760MB
%chk=meoh_532.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4216 0.1040 0.0270
C 0.0346 -0.0030 0.0032
H 1.7280 -0.6460 -0.5244
H -0.3591 -0.7275 0.7161
H -0.5064 0.9099 0.2523
H -0.3804 -0.3671 -0.9366

