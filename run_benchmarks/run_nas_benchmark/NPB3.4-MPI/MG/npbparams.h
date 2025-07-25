! CLASS = C
!  
!  
!  This file is generated automatically by the setparams utility.
!  It sets the number of processors and the class of the NPB
!  in this directory. Do not modify it by hand.
!  
        integer nx_default, ny_default, nz_default
        parameter (nx_default=512, ny_default=512, nz_default=512)
        integer nit_default, lt_default
        parameter (nit_default=20, lt_default=9)
        integer debug_default
        parameter (debug_default=0)
        logical  convertdouble
        parameter (convertdouble = .false.)
        character*11 compiletime
        parameter (compiletime='09 Jul 2025')
        character*5 npbversion
        parameter (npbversion='3.4.3')
        character*6 cs1
        parameter (cs1='mpif90')
        character*8 cs2
        parameter (cs2='$(MPIFC)')
        character*45 cs3
        parameter (cs3='-L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi')
        character*44 cs4
        parameter (cs4='-I/usr/lib/x86_64-linux-gnu/openmpi/include/')
        character*3 cs5
        parameter (cs5='-O3')
        character*9 cs6
        parameter (cs6='$(FFLAGS)')
        character*6 cs7
        parameter (cs7='randi8')
