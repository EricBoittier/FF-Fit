%nproc=4
%mem=5760MB
%chk=meoh_314.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4174 0.0268 0.0475
C 0.0388 -0.0115 0.0100
H 1.7067 0.4391 -0.7932
H -0.3159 0.3295 -0.9626
H -0.4655 -0.9662 0.1592
H -0.4278 0.6932 0.6982

