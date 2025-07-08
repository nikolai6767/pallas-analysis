! CLASS = C
!  
!  
!  This file is generated automatically by the setparams utility.
!  It sets the number of processors and the class of the NPB
!  in this directory. Do not modify it by hand.
!  
        integer            na, nonzer, niter
        double precision   shift, rcond
        parameter(  na=150000, &
     &              nonzer=15, &
     &              niter=75, &
     &              shift=110., &
     &              rcond=1.0d-1 )
        logical  convertdouble
        parameter (convertdouble = .false.)
        character*11 compiletime
        parameter (compiletime='08 Jul 2025')
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
