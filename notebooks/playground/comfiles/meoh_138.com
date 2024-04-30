%nproc=4
%mem=5760MB
%chk=meoh_138.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4293 0.0107 0.0439
C 0.0225 -0.0007 0.0032
H 1.7041 0.6215 -0.6716
H -0.3830 -0.9776 -0.2603
H -0.4592 0.3026 0.9327
H -0.3585 0.6774 -0.7604

