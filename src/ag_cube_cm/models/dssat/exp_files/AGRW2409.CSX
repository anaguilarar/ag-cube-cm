*EXP.DETAILS: IINA1500CS HISTORICAL_PLANTA

*GENERAL
@PEOPLE
-99
@ADDRESS
-99
@SITE
-99
@ PAREA  PRNO  PLEN  PLDR  PLSP  PLAY HAREA  HRNO  HLEN  HARM.........
    -99   -99   -99   -99   -99   -99   -99   -99   -99   -99
@NOTES
 -99


*TREATMENTS                        -------------FACTOR LEVELS------------
@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM
 1 1 0 0 0N_CP_1Jun                 1  1  1  1  1  0  0  0  0  0  0  1  1

*CULTIVARS
@C CR INGENO CNAME
 1 CS UC0006 -99

*FIELDS
@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL    FLNAME
 1 CIAT0001 HUKK8332   -99   -99   -99   -99   -99   -99 -99    209  IB00000101 -99
@L ...........XCRD ...........YCRD .....ELEV .............AREA .SLEN .FLWR .SLAS FLHST FHDUR
 1               0               0         0                 0     0     0     0   -99   -99

*SOIL ANALYSIS
@A SADAT  SMHB  SMPX  SMKE  SANAME
 1 83151   -99 SA002   -99  -99
@A  SABL  SADM  SAOC  SANI SAPHW SAPHB  SAPX  SAKE  SASC
 1    19   -99  1.56     0   4.8   -99   1.4     0     0
 1    65   -99   .86     0   4.8   -99     1     0     0
 1   127   -99   .86     0   4.8   -99    .8     0     0
 1   183   -99   .26     0   4.8   -99    .2     0     0

*INITIAL CONDITIONS
@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME
 1    CS 83151   -99   -99     0     0   -99   -99  0.10  0.10     0   -99 DEFAUL
@C  ICBL  SH2O  SNH4  SNO3
 1    20   .15    .6   1.2
 1    40   .19    .6   1.2
 1    60   .17    .4   1.2
 1    90   .07    .4   1.2
 1   150   .17    .4   0.6
 1   200   .07    .2   0.6

*PLANTING DETAILS
@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  SPRL                        PLNAME
 1 83151   -99     1     1     V     R   100    90    20   450   -99   -99   1.5    25                        1-Jun

*FERTILIZERS (INORGANIC)
@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME
 1 86269 FE006   -99     4   -99   -99   -99   -99   -99   -99 Nofertilizer

*HARVEST DETAILS
@H HDATE  HSTG  HCOM HSIZE   HPC  HBPC HNAME
 1 83365 GS000   -99   -99   -99   -99 H1

*SIMULATION CONTROLS
@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL
 1 GE             23     1     S 83001  2150 DEFAULT                   CSYCA
@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2
 1 OP              N     N     N     N     N     N     N     N     M
@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL
 1 ME              M     M     E     H     S     V     R     1     G     R     2
@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS
 1 MA              R     N     R     N     R
@N OUTPUTS     FNAME OVVEW SUMRY FROPT GROUT CAOUT WAOUT NIOUT MIOUT DIOUT VBOSE CHOUT OPOUT FMOPT
 1 OU              N     N     Y     1     N     N     N     N     N     N     0     N     N     A

@  AUTOMATIC MANAGEMENT
@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN
 1 PL          83187 83201    40   100    30    40    10
@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF
 1 IR             30    50   100 IB001 IR011    10     1
@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF
 1 NI             30    50    25 IB001 IB001
@N RESIDUES    RIPCN RTIME RIDEP
 1 RE            100     1    20
@N HARVEST     HFRST HLAST HPCNP HPCNR
 1 HA              0 80194   100     0



