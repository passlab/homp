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

set object 15 rect from 0.098217, 0 to 161.466239, 10 fc rgb "#FF0000"
set object 16 rect from 0.081436, 0 to 132.576257, 10 fc rgb "#00FF00"
set object 17 rect from 0.083108, 0 to 133.383204, 10 fc rgb "#0000FF"
set object 18 rect from 0.083408, 0 to 134.260599, 10 fc rgb "#FFFF00"
set object 19 rect from 0.083992, 0 to 135.655144, 10 fc rgb "#FF00FF"
set object 20 rect from 0.084837, 0 to 137.088115, 10 fc rgb "#808080"
set object 21 rect from 0.085732, 0 to 137.845429, 10 fc rgb "#800080"
set object 22 rect from 0.086413, 0 to 138.567518, 10 fc rgb "#008080"
set object 23 rect from 0.086649, 0 to 156.506076, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.096934, 10 to 159.829929, 20 fc rgb "#FF0000"
set object 25 rect from 0.069750, 10 to 113.872380, 20 fc rgb "#00FF00"
set object 26 rect from 0.071440, 10 to 114.797808, 20 fc rgb "#0000FF"
set object 27 rect from 0.071814, 10 to 115.793683, 20 fc rgb "#FFFF00"
set object 28 rect from 0.072447, 10 to 117.389965, 20 fc rgb "#FF00FF"
set object 29 rect from 0.073444, 10 to 118.862963, 20 fc rgb "#808080"
set object 30 rect from 0.074375, 10 to 119.652298, 20 fc rgb "#800080"
set object 31 rect from 0.075079, 10 to 120.497671, 20 fc rgb "#008080"
set object 32 rect from 0.075391, 10 to 154.378234, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.095527, 20 to 157.076063, 30 fc rgb "#FF0000"
set object 34 rect from 0.055214, 20 to 90.240363, 30 fc rgb "#00FF00"
set object 35 rect from 0.056676, 20 to 91.020091, 30 fc rgb "#0000FF"
set object 36 rect from 0.056974, 20 to 92.009562, 30 fc rgb "#FFFF00"
set object 37 rect from 0.057600, 20 to 93.554610, 30 fc rgb "#FF00FF"
set object 38 rect from 0.058530, 20 to 94.710593, 30 fc rgb "#808080"
set object 39 rect from 0.059258, 20 to 95.451895, 30 fc rgb "#800080"
set object 40 rect from 0.059943, 20 to 96.287661, 30 fc rgb "#008080"
set object 41 rect from 0.060240, 20 to 152.269605, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.094292, 30 to 155.284449, 40 fc rgb "#FF0000"
set object 43 rect from 0.043095, 30 to 71.060964, 40 fc rgb "#00FF00"
set object 44 rect from 0.044704, 30 to 71.851900, 40 fc rgb "#0000FF"
set object 45 rect from 0.044985, 30 to 72.947042, 40 fc rgb "#FFFF00"
set object 46 rect from 0.045662, 30 to 74.416838, 40 fc rgb "#FF00FF"
set object 47 rect from 0.046588, 30 to 75.638466, 40 fc rgb "#808080"
set object 48 rect from 0.047344, 30 to 76.406987, 40 fc rgb "#800080"
set object 49 rect from 0.048042, 30 to 77.261966, 40 fc rgb "#008080"
set object 50 rect from 0.048372, 30 to 150.192998, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.092978, 40 to 153.488030, 50 fc rgb "#FF0000"
set object 52 rect from 0.030738, 40 to 51.833532, 50 fc rgb "#00FF00"
set object 53 rect from 0.032704, 40 to 52.715729, 50 fc rgb "#0000FF"
set object 54 rect from 0.033048, 40 to 53.826883, 50 fc rgb "#FFFF00"
set object 55 rect from 0.033733, 40 to 55.429569, 50 fc rgb "#FF00FF"
set object 56 rect from 0.034740, 40 to 56.777683, 50 fc rgb "#808080"
set object 57 rect from 0.035575, 40 to 57.555810, 50 fc rgb "#800080"
set object 58 rect from 0.036294, 40 to 58.489243, 50 fc rgb "#008080"
set object 59 rect from 0.036633, 40 to 148.039538, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.091529, 50 to 154.050010, 60 fc rgb "#FF0000"
set object 61 rect from 0.014415, 50 to 26.730117, 60 fc rgb "#00FF00"
set object 62 rect from 0.017106, 50 to 28.238339, 60 fc rgb "#0000FF"
set object 63 rect from 0.017750, 50 to 30.582328, 60 fc rgb "#FFFF00"
set object 64 rect from 0.019247, 50 to 32.945530, 60 fc rgb "#FF00FF"
set object 65 rect from 0.020681, 50 to 34.492178, 60 fc rgb "#808080"
set object 66 rect from 0.021671, 50 to 35.515271, 60 fc rgb "#800080"
set object 67 rect from 0.022673, 50 to 37.151581, 60 fc rgb "#008080"
set object 68 rect from 0.023327, 50 to 145.322497, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
