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

set object 15 rect from 0.116067, 0 to 160.985539, 10 fc rgb "#FF0000"
set object 16 rect from 0.096852, 0 to 133.372695, 10 fc rgb "#00FF00"
set object 17 rect from 0.098742, 0 to 134.215907, 10 fc rgb "#0000FF"
set object 18 rect from 0.099137, 0 to 135.259842, 10 fc rgb "#FFFF00"
set object 19 rect from 0.099891, 0 to 136.650759, 10 fc rgb "#FF00FF"
set object 20 rect from 0.100918, 0 to 137.275714, 10 fc rgb "#808080"
set object 21 rect from 0.101385, 0 to 138.018646, 10 fc rgb "#800080"
set object 22 rect from 0.102251, 0 to 138.851030, 10 fc rgb "#008080"
set object 23 rect from 0.102550, 0 to 156.690772, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.114574, 10 to 159.320851, 20 fc rgb "#FF0000"
set object 25 rect from 0.081498, 10 to 112.672415, 20 fc rgb "#00FF00"
set object 26 rect from 0.083493, 10 to 113.614696, 20 fc rgb "#0000FF"
set object 27 rect from 0.083925, 10 to 114.729013, 20 fc rgb "#FFFF00"
set object 28 rect from 0.084777, 10 to 116.303037, 20 fc rgb "#FF00FF"
set object 29 rect from 0.085947, 10 to 116.957809, 20 fc rgb "#808080"
set object 30 rect from 0.086410, 10 to 117.791567, 20 fc rgb "#800080"
set object 31 rect from 0.087270, 10 to 118.618538, 20 fc rgb "#008080"
set object 32 rect from 0.087620, 10 to 154.533812, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.113005, 20 to 156.772143, 30 fc rgb "#FF0000"
set object 34 rect from 0.070062, 20 to 96.782285, 30 fc rgb "#00FF00"
set object 35 rect from 0.071777, 20 to 97.629537, 30 fc rgb "#0000FF"
set object 36 rect from 0.072134, 20 to 98.682927, 30 fc rgb "#FFFF00"
set object 37 rect from 0.072930, 20 to 100.147135, 30 fc rgb "#FF00FF"
set object 38 rect from 0.074011, 20 to 100.682637, 30 fc rgb "#808080"
set object 39 rect from 0.074413, 20 to 101.455387, 30 fc rgb "#800080"
set object 40 rect from 0.075233, 20 to 102.320336, 30 fc rgb "#008080"
set object 41 rect from 0.075596, 20 to 152.478588, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.111412, 30 to 154.867219, 40 fc rgb "#FF0000"
set object 43 rect from 0.055899, 30 to 77.788874, 40 fc rgb "#00FF00"
set object 44 rect from 0.057773, 30 to 78.602269, 40 fc rgb "#0000FF"
set object 45 rect from 0.058102, 30 to 79.837310, 40 fc rgb "#FFFF00"
set object 46 rect from 0.059010, 30 to 81.293357, 40 fc rgb "#FF00FF"
set object 47 rect from 0.060105, 30 to 81.874919, 40 fc rgb "#808080"
set object 48 rect from 0.060539, 30 to 82.678859, 40 fc rgb "#800080"
set object 49 rect from 0.061399, 30 to 83.535647, 40 fc rgb "#008080"
set object 50 rect from 0.061757, 30 to 150.298599, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.109878, 40 to 152.813449, 50 fc rgb "#FF0000"
set object 52 rect from 0.042340, 40 to 59.840691, 50 fc rgb "#00FF00"
set object 53 rect from 0.044549, 40 to 60.736831, 50 fc rgb "#0000FF"
set object 54 rect from 0.044960, 40 to 61.921692, 50 fc rgb "#FFFF00"
set object 55 rect from 0.045819, 40 to 63.505170, 50 fc rgb "#FF00FF"
set object 56 rect from 0.046986, 40 to 64.113883, 50 fc rgb "#808080"
set object 57 rect from 0.047431, 40 to 64.877178, 50 fc rgb "#800080"
set object 58 rect from 0.048283, 40 to 65.824792, 50 fc rgb "#008080"
set object 59 rect from 0.048708, 40 to 148.100994, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.108169, 50 to 154.262708, 60 fc rgb "#FF0000"
set object 61 rect from 0.022536, 50 to 33.945463, 60 fc rgb "#00FF00"
set object 62 rect from 0.025509, 50 to 35.501790, 60 fc rgb "#0000FF"
set object 63 rect from 0.026319, 50 to 38.195544, 60 fc rgb "#FFFF00"
set object 64 rect from 0.028367, 50 to 40.615528, 60 fc rgb "#FF00FF"
set object 65 rect from 0.030108, 50 to 41.620029, 60 fc rgb "#808080"
set object 66 rect from 0.030855, 50 to 42.872767, 60 fc rgb "#800080"
set object 67 rect from 0.032245, 50 to 44.404691, 60 fc rgb "#008080"
set object 68 rect from 0.032901, 50 to 145.351645, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
