%nproc=4
%mem=5760MB
%chk=meoh_857.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4341 -0.0097 -0.0130
C 0.0184 0.0024 -0.0082
H 1.6670 0.9303 0.1374
H -0.4982 0.1199 -0.9609
H -0.3447 -0.9212 0.4427
H -0.3464 0.7821 0.6603

