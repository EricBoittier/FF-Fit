%nproc=4
%mem=5760MB
%chk=meoh_427.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4177 0.1056 0.0003
C 0.0266 0.0043 0.0138
H 1.8142 -0.7641 -0.2161
H -0.4657 0.6317 0.7569
H -0.4167 0.2049 -0.9616
H -0.2923 -1.0159 0.2272

