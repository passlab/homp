set title "Offloading (axpy_kernel) Profile on 6 Devices"
set yrange [0:72.000000]
set xlabel "execution time in ms"
set xrange [0:158.400000]
set style fill pattern 2 bo 1
set style rect fs solid 1 noborder
set border 15 lw 0.2
set xtics out nomirror
unset key
set ytics out nomirror ("dev 0(sysid:0,type:HOSTCPU)" 5,"dev 1(sysid:1,type:HOSTCPU)" 15,"dev 2(sysid:0,type:THSIM)" 25,"dev 3(sysid:1,type:THSIM)" 35,"dev 4(sysid:2,type:THSIM)" 45,"dev 5(sysid:3,type:THSIM)" 55)
set object 1 rect from 4, 65 to 17, 68 fc rgb "#FF0000"
set label "ACCU_TOTAL" at 4,63 font "Helvetica,8'"

set object 2 rect from 21, 65 to 34, 68 fc rgb "#00FF00"
set label "INIT_0" at 21,63 font "Helvetica,8'"

set object 3 rect from 38, 65 to 51, 68 fc rgb "#0000FF"
set label "INIT_0.1" at 38,63 font "Helvetica,8'"

set object 4 rect from 55, 65 to 68, 68 fc rgb "#FFFF00"
set label "INIT_1" at 55,63 font "Helvetica,8'"

set object 5 rect from 72, 65 to 85, 68 fc rgb "#00FFFF"
set label "MODELING" at 72,63 font "Helvetica,8'"

set object 6 rect from 89, 65 to 102, 68 fc rgb "#FF00FF"
set label "ACC_MAPTO" at 89,63 font "Helvetica,8'"

set object 7 rect from 106, 65 to 119, 68 fc rgb "#808080"
set label "KERN" at 106,63 font "Helvetica,8'"

set object 8 rect from 123, 65 to 136, 68 fc rgb "#800000"
set label "PRE_BAR_X" at 123,63 font "Helvetica,8'"

set object 9 rect from 140, 65 to 153, 68 fc rgb "#808000"
set label "DATA_X" at 140,63 font "Helvetica,8'"

set object 10 rect from 157, 65 to 170, 68 fc rgb "#008000"
set label "POST_BAR_X" at 157,63 font "Helvetica,8'"

set object 11 rect from 174, 65 to 187, 68 fc rgb "#800080"
set label "ACC_MAPFROM" at 174,63 font "Helvetica,8'"

set object 12 rect from 191, 65 to 204, 68 fc rgb "#008080"
set label "FINI_1" at 191,63 font "Helvetica,8'"

set object 13 rect from 208, 65 to 221, 68 fc rgb "#000080"
set label "BAR_FINI_2" at 208,63 font "Helvetica,8'"

set object 14 rect from 225, 65 to 238, 68 fc rgb "(null)"
set label "PROF_BAR" at 225,63 font "Helvetica,8'"

set object 15 rect from 0.132357, 0 to 160.672241, 10 fc rgb "#FF0000"
set object 16 rect from 0.110461, 0 to 132.870588, 10 fc rgb "#00FF00"
set object 17 rect from 0.112700, 0 to 133.630997, 10 fc rgb "#0000FF"
set object 18 rect from 0.113113, 0 to 134.529771, 10 fc rgb "#FFFF00"
set object 19 rect from 0.113871, 0 to 135.707637, 10 fc rgb "#FF00FF"
set object 20 rect from 0.114876, 0 to 137.221359, 10 fc rgb "#808080"
set object 21 rect from 0.116156, 0 to 137.850501, 10 fc rgb "#800080"
set object 22 rect from 0.116937, 0 to 138.458355, 10 fc rgb "#008080"
set object 23 rect from 0.117199, 0 to 155.883909, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.130833, 10 to 159.399768, 20 fc rgb "#FF0000"
set object 25 rect from 0.094422, 10 to 113.885190, 20 fc rgb "#00FF00"
set object 26 rect from 0.096646, 10 to 114.737842, 20 fc rgb "#0000FF"
set object 27 rect from 0.097137, 10 to 115.678006, 20 fc rgb "#FFFF00"
set object 28 rect from 0.097969, 10 to 117.150338, 20 fc rgb "#FF00FF"
set object 29 rect from 0.099189, 10 to 118.679434, 20 fc rgb "#808080"
set object 30 rect from 0.100493, 10 to 119.467043, 20 fc rgb "#800080"
set object 31 rect from 0.101415, 10 to 120.171870, 20 fc rgb "#008080"
set object 32 rect from 0.101736, 10 to 153.992938, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.129093, 20 to 157.305390, 30 fc rgb "#FF0000"
set object 34 rect from 0.079503, 20 to 96.056370, 30 fc rgb "#00FF00"
set object 35 rect from 0.081617, 20 to 96.958691, 30 fc rgb "#0000FF"
set object 36 rect from 0.082101, 20 to 97.936699, 30 fc rgb "#FFFF00"
set object 37 rect from 0.082930, 20 to 99.463430, 30 fc rgb "#FF00FF"
set object 38 rect from 0.084219, 20 to 100.842337, 30 fc rgb "#808080"
set object 39 rect from 0.085396, 20 to 101.577912, 30 fc rgb "#800080"
set object 40 rect from 0.086260, 20 to 102.294565, 30 fc rgb "#008080"
set object 41 rect from 0.086637, 20 to 152.086593, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.127532, 30 to 155.493651, 40 fc rgb "#FF0000"
set object 43 rect from 0.063814, 30 to 77.557019, 40 fc rgb "#00FF00"
set object 44 rect from 0.066047, 30 to 78.408489, 40 fc rgb "#0000FF"
set object 45 rect from 0.066434, 30 to 79.540233, 40 fc rgb "#FFFF00"
set object 46 rect from 0.067376, 30 to 81.074060, 40 fc rgb "#FF00FF"
set object 47 rect from 0.068670, 30 to 82.451784, 40 fc rgb "#808080"
set object 48 rect from 0.069849, 30 to 83.222837, 40 fc rgb "#800080"
set object 49 rect from 0.070750, 30 to 83.970238, 40 fc rgb "#008080"
set object 50 rect from 0.071125, 30 to 150.188526, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.125951, 40 to 153.770609, 50 fc rgb "#FF0000"
set object 52 rect from 0.048319, 40 to 59.797973, 50 fc rgb "#00FF00"
set object 53 rect from 0.050937, 40 to 60.615146, 50 fc rgb "#0000FF"
set object 54 rect from 0.051375, 40 to 61.671205, 50 fc rgb "#FFFF00"
set object 55 rect from 0.052288, 40 to 63.223954, 50 fc rgb "#FF00FF"
set object 56 rect from 0.053606, 40 to 64.781433, 50 fc rgb "#808080"
set object 57 rect from 0.054902, 40 to 65.546573, 50 fc rgb "#800080"
set object 58 rect from 0.055854, 40 to 66.380303, 50 fc rgb "#008080"
set object 59 rect from 0.056280, 40 to 148.217139, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.123890, 50 to 154.379646, 60 fc rgb "#FF0000"
set object 61 rect from 0.026495, 50 to 34.736396, 60 fc rgb "#00FF00"
set object 62 rect from 0.029892, 50 to 36.474813, 60 fc rgb "#0000FF"
set object 63 rect from 0.030962, 50 to 38.923969, 60 fc rgb "#FFFF00"
set object 64 rect from 0.033135, 50 to 41.124780, 60 fc rgb "#FF00FF"
set object 65 rect from 0.034894, 50 to 42.869110, 60 fc rgb "#808080"
set object 66 rect from 0.036397, 50 to 43.769065, 60 fc rgb "#800080"
set object 67 rect from 0.037625, 50 to 45.196459, 60 fc rgb "#008080"
set object 68 rect from 0.038374, 50 to 145.293761, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
