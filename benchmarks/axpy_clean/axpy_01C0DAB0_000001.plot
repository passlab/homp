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

set object 15 rect from 0.104023, 0 to 160.155826, 10 fc rgb "#FF0000"
set object 16 rect from 0.087061, 0 to 133.140045, 10 fc rgb "#00FF00"
set object 17 rect from 0.088622, 0 to 133.991766, 10 fc rgb "#0000FF"
set object 18 rect from 0.088986, 0 to 134.837466, 10 fc rgb "#FFFF00"
set object 19 rect from 0.089545, 0 to 136.106826, 10 fc rgb "#FF00FF"
set object 20 rect from 0.090388, 0 to 136.730948, 10 fc rgb "#808080"
set object 21 rect from 0.090810, 0 to 137.407814, 10 fc rgb "#800080"
set object 22 rect from 0.091476, 0 to 138.081624, 10 fc rgb "#008080"
set object 23 rect from 0.091704, 0 to 156.106671, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.102721, 10 to 158.477903, 20 fc rgb "#FF0000"
set object 25 rect from 0.075130, 10 to 115.200899, 20 fc rgb "#00FF00"
set object 26 rect from 0.076718, 10 to 116.033031, 20 fc rgb "#0000FF"
set object 27 rect from 0.077071, 10 to 116.930038, 20 fc rgb "#FFFF00"
set object 28 rect from 0.077674, 10 to 118.466172, 20 fc rgb "#FF00FF"
set object 29 rect from 0.078694, 10 to 119.037460, 20 fc rgb "#808080"
set object 30 rect from 0.079073, 10 to 119.770125, 20 fc rgb "#800080"
set object 31 rect from 0.079774, 10 to 120.529926, 20 fc rgb "#008080"
set object 32 rect from 0.080056, 10 to 154.074636, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.101377, 20 to 156.258973, 30 fc rgb "#FF0000"
set object 34 rect from 0.063843, 20 to 97.923612, 30 fc rgb "#00FF00"
set object 35 rect from 0.065264, 20 to 98.799413, 30 fc rgb "#0000FF"
set object 36 rect from 0.065638, 20 to 99.788340, 30 fc rgb "#FFFF00"
set object 37 rect from 0.066295, 20 to 101.116464, 30 fc rgb "#FF00FF"
set object 38 rect from 0.067174, 20 to 101.604907, 30 fc rgb "#808080"
set object 39 rect from 0.067501, 20 to 102.268205, 30 fc rgb "#800080"
set object 40 rect from 0.068157, 20 to 103.026478, 30 fc rgb "#008080"
set object 41 rect from 0.068445, 20 to 152.161566, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.100064, 30 to 154.544929, 40 fc rgb "#FF0000"
set object 43 rect from 0.051550, 30 to 79.901531, 40 fc rgb "#00FF00"
set object 44 rect from 0.053304, 30 to 80.635723, 40 fc rgb "#0000FF"
set object 45 rect from 0.053590, 30 to 81.763294, 40 fc rgb "#FFFF00"
set object 46 rect from 0.054339, 30 to 83.263217, 40 fc rgb "#FF00FF"
set object 47 rect from 0.055354, 30 to 83.828575, 40 fc rgb "#808080"
set object 48 rect from 0.055709, 30 to 84.541562, 40 fc rgb "#800080"
set object 49 rect from 0.056414, 30 to 85.326970, 40 fc rgb "#008080"
set object 50 rect from 0.056710, 30 to 150.115963, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.098732, 40 to 152.527899, 50 fc rgb "#FF0000"
set object 52 rect from 0.039096, 40 to 61.330804, 50 fc rgb "#00FF00"
set object 53 rect from 0.040989, 40 to 62.162937, 50 fc rgb "#0000FF"
set object 54 rect from 0.041340, 40 to 63.258879, 50 fc rgb "#FFFF00"
set object 55 rect from 0.042077, 40 to 64.668320, 50 fc rgb "#FF00FF"
set object 56 rect from 0.043019, 40 to 65.230623, 50 fc rgb "#808080"
set object 57 rect from 0.043381, 40 to 66.007046, 50 fc rgb "#800080"
set object 58 rect from 0.044132, 40 to 66.861732, 50 fc rgb "#008080"
set object 59 rect from 0.044473, 40 to 147.996411, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.097153, 50 to 153.522936, 60 fc rgb "#FF0000"
set object 61 rect from 0.021321, 50 to 35.531549, 60 fc rgb "#00FF00"
set object 62 rect from 0.023990, 50 to 37.058698, 60 fc rgb "#0000FF"
set object 63 rect from 0.024693, 50 to 39.723930, 60 fc rgb "#FFFF00"
set object 64 rect from 0.026495, 50 to 42.123826, 60 fc rgb "#FF00FF"
set object 65 rect from 0.028071, 50 to 42.904742, 60 fc rgb "#808080"
set object 66 rect from 0.028598, 50 to 43.846945, 60 fc rgb "#800080"
set object 67 rect from 0.029557, 50 to 45.254858, 60 fc rgb "#008080"
set object 68 rect from 0.030138, 50 to 145.380867, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
