%nproc=4
%mem=5760MB
%chk=meoh_675.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4368 0.0015 0.0205
C -0.0115 0.0028 0.0169
H 1.7249 0.7800 -0.5004
H -0.2670 -0.9985 -0.3296
H -0.3684 0.2210 1.0235
H -0.2993 0.7254 -0.7467

