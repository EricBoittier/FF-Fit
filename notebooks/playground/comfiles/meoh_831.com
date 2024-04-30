%nproc=4
%mem=5760MB
%chk=meoh_831.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4403 0.0319 -0.0682
C -0.0079 0.0010 0.0101
H 1.6899 0.3471 0.8255
H -0.3956 -0.1963 -0.9893
H -0.2520 -0.8312 0.6704
H -0.3501 0.9465 0.4309

