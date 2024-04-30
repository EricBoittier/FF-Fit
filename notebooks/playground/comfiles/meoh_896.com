%nproc=4
%mem=5760MB
%chk=meoh_896.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4338 0.0647 0.0503
C -0.0212 -0.0002 0.0008
H 1.8181 -0.0972 -0.8366
H -0.3405 0.5580 -0.8793
H -0.2178 -1.0673 -0.1033
H -0.3061 0.3674 0.9866

