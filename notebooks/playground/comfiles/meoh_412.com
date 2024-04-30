%nproc=4
%mem=5760MB
%chk=meoh_412.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4501 0.0675 0.0634
C -0.0181 -0.0200 -0.0082
H 1.5851 -0.0916 -0.8942
H -0.3262 0.9007 0.4874
H -0.2788 0.0550 -1.0639
H -0.3236 -0.9127 0.5375

