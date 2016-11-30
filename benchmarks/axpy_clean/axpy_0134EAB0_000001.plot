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

set object 15 rect from 82.289181, 0 to 160.048061, 10 fc rgb "#FF0000"
set object 16 rect from 23.147616, 0 to 40.516027, 10 fc rgb "#00FF00"
set object 17 rect from 23.151212, 0 to 40.517967, 10 fc rgb "#0000FF"
set object 18 rect from 23.152011, 0 to 40.520268, 10 fc rgb "#FFFF00"
set object 19 rect from 23.153368, 0 to 40.522841, 10 fc rgb "#FF00FF"
set object 20 rect from 23.154813, 0 to 56.550277, 10 fc rgb "#808080"
set object 21 rect from 32.313528, 0 to 56.553016, 10 fc rgb "#800080"
set object 22 rect from 32.314887, 0 to 56.554456, 10 fc rgb "#008080"
set object 23 rect from 32.315227, 0 to 144.012203, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 82.285976, 10 to 195.411997, 20 fc rgb "#FF0000"
set object 25 rect from 0.040007, 10 to 0.077365, 20 fc rgb "#00FF00"
set object 26 rect from 0.044888, 10 to 0.080354, 20 fc rgb "#0000FF"
set object 27 rect from 0.046045, 10 to 0.083443, 20 fc rgb "#FFFF00"
set object 28 rect from 0.047839, 10 to 0.086203, 20 fc rgb "#FF00FF"
set object 29 rect from 0.049387, 10 to 51.480676, 20 fc rgb "#808080"
set object 30 rect from 29.416419, 10 to 51.483643, 20 fc rgb "#800080"
set object 31 rect from 29.418542, 10 to 51.485709, 20 fc rgb "#008080"
set object 32 rect from 29.419021, 10 to 144.005336, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 82.300868, 20 to 230.506749, 30 fc rgb "#FF0000"
set object 34 rect from 32.687881, 20 to 57.214507, 30 fc rgb "#00FF00"
set object 35 rect from 32.692733, 20 to 57.216630, 30 fc rgb "#0000FF"
set object 36 rect from 32.693609, 20 to 57.259553, 30 fc rgb "#FFFF00"
set object 37 rect from 32.718156, 20 to 133.568592, 30 fc rgb "#FF00FF"
set object 38 rect from 76.321116, 20 to 143.026830, 30 fc rgb "#808080"
set object 39 rect from 81.725747, 20 to 143.689652, 30 fc rgb "#800080"
set object 40 rect from 82.105062, 20 to 143.976473, 30 fc rgb "#008080"
set object 41 rect from 82.268142, 20 to 144.033080, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 82.295766, 30 to 244.155399, 40 fc rgb "#FF0000"
set object 43 rect from 18.017774, 30 to 31.539381, 40 fc rgb "#00FF00"
set object 44 rect from 18.021959, 30 to 31.541691, 40 fc rgb "#0000FF"
set object 45 rect from 18.022980, 30 to 31.565562, 40 fc rgb "#FFFF00"
set object 46 rect from 18.036657, 30 to 83.566615, 40 fc rgb "#FF00FF"
set object 47 rect from 47.749933, 30 to 105.845049, 40 fc rgb "#808080"
set object 48 rect from 60.480495, 30 to 131.672259, 40 fc rgb "#800080"
set object 49 rect from 75.238284, 30 to 132.069081, 40 fc rgb "#008080"
set object 50 rect from 75.464307, 30 to 144.023709, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 82.298984, 40 to 206.364699, 50 fc rgb "#FF0000"
set object 52 rect from 35.099210, 40 to 61.431429, 50 fc rgb "#00FF00"
set object 53 rect from 35.102294, 40 to 61.433272, 50 fc rgb "#0000FF"
set object 54 rect from 35.102970, 40 to 65.745964, 50 fc rgb "#FFFF00"
set object 55 rect from 37.567431, 40 to 90.924363, 50 fc rgb "#FF00FF"
set object 56 rect from 51.954106, 40 to 123.218232, 50 fc rgb "#808080"
set object 57 rect from 70.407023, 40 to 123.767538, 50 fc rgb "#800080"
set object 58 rect from 70.721800, 40 to 132.474439, 50 fc rgb "#008080"
set object 59 rect from 75.695997, 40 to 144.029375, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 82.292617, 50 to 243.409747, 60 fc rgb "#FF0000"
set object 61 rect from 9.609861, 50 to 16.824196, 60 fc rgb "#00FF00"
set object 62 rect from 9.613827, 50 to 16.826911, 60 fc rgb "#0000FF"
set object 63 rect from 9.615002, 50 to 16.844466, 60 fc rgb "#FFFF00"
set object 64 rect from 9.625072, 50 to 75.644726, 60 fc rgb "#FF00FF"
set object 65 rect from 43.223375, 50 to 115.552656, 60 fc rgb "#808080"
set object 66 rect from 66.027079, 50 to 116.216577, 60 fc rgb "#800080"
set object 67 rect from 66.407304, 50 to 131.212326, 60 fc rgb "#008080"
set object 68 rect from 74.975023, 50 to 144.018173, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
