%nproc=4
%mem=5760MB
%chk=meoh_767.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4326 0.1133 0.0056
C 0.0008 -0.0063 0.0095
H 1.7631 -0.7842 -0.2080
H -0.2933 -0.6799 -0.7954
H -0.2890 -0.4309 0.9707
H -0.4699 0.9554 -0.1949

