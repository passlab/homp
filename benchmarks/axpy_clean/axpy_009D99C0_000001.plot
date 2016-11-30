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

set object 15 rect from 0.124187, 0 to 154.019528, 10 fc rgb "#FF0000"
set object 16 rect from 0.047359, 0 to 59.499286, 10 fc rgb "#00FF00"
set object 17 rect from 0.049964, 0 to 60.409932, 10 fc rgb "#0000FF"
set object 18 rect from 0.050475, 0 to 61.523346, 10 fc rgb "#FFFF00"
set object 19 rect from 0.051430, 0 to 63.125082, 10 fc rgb "#FF00FF"
set object 20 rect from 0.052769, 0 to 64.717212, 10 fc rgb "#808080"
set object 21 rect from 0.054114, 0 to 65.546275, 10 fc rgb "#800080"
set object 22 rect from 0.055055, 0 to 66.322547, 10 fc rgb "#008080"
set object 23 rect from 0.055419, 0 to 148.201700, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.129156, 10 to 159.327437, 20 fc rgb "#FF0000"
set object 25 rect from 0.093392, 10 to 114.236562, 20 fc rgb "#00FF00"
set object 26 rect from 0.095564, 10 to 115.020030, 20 fc rgb "#0000FF"
set object 27 rect from 0.095995, 10 to 115.889886, 20 fc rgb "#FFFF00"
set object 28 rect from 0.096719, 10 to 117.262457, 20 fc rgb "#FF00FF"
set object 29 rect from 0.097870, 10 to 118.811398, 20 fc rgb "#808080"
set object 30 rect from 0.099156, 10 to 119.513282, 20 fc rgb "#800080"
set object 31 rect from 0.100007, 10 to 120.197167, 20 fc rgb "#008080"
set object 32 rect from 0.100299, 10 to 154.249891, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.122247, 20 to 154.846195, 30 fc rgb "#FF0000"
set object 34 rect from 0.025065, 20 to 33.788768, 30 fc rgb "#00FF00"
set object 35 rect from 0.029015, 20 to 36.201567, 30 fc rgb "#0000FF"
set object 36 rect from 0.030304, 20 to 38.463191, 30 fc rgb "#FFFF00"
set object 37 rect from 0.032224, 20 to 40.681620, 30 fc rgb "#FF00FF"
set object 38 rect from 0.034056, 20 to 42.522114, 30 fc rgb "#808080"
set object 39 rect from 0.035596, 20 to 43.511947, 30 fc rgb "#800080"
set object 40 rect from 0.036842, 20 to 44.942109, 30 fc rgb "#008080"
set object 41 rect from 0.037601, 20 to 145.325783, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.127483, 30 to 157.381372, 40 fc rgb "#FF0000"
set object 43 rect from 0.079687, 30 to 97.703315, 40 fc rgb "#00FF00"
set object 44 rect from 0.081804, 30 to 98.491588, 40 fc rgb "#0000FF"
set object 45 rect from 0.082207, 30 to 99.504217, 40 fc rgb "#FFFF00"
set object 46 rect from 0.083050, 30 to 100.829995, 40 fc rgb "#FF00FF"
set object 47 rect from 0.084158, 30 to 102.268555, 40 fc rgb "#808080"
set object 48 rect from 0.085357, 30 to 103.006433, 40 fc rgb "#800080"
set object 49 rect from 0.086235, 30 to 103.793503, 40 fc rgb "#008080"
set object 50 rect from 0.086629, 30 to 152.336213, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.130713, 40 to 161.274722, 50 fc rgb "#FF0000"
set object 52 rect from 0.108217, 40 to 131.922814, 50 fc rgb "#00FF00"
set object 53 rect from 0.110297, 40 to 132.657094, 50 fc rgb "#0000FF"
set object 54 rect from 0.110679, 40 to 133.681719, 50 fc rgb "#FFFF00"
set object 55 rect from 0.111531, 40 to 135.043496, 50 fc rgb "#FF00FF"
set object 56 rect from 0.112671, 40 to 136.464056, 50 fc rgb "#808080"
set object 57 rect from 0.113857, 40 to 137.192338, 50 fc rgb "#800080"
set object 58 rect from 0.114745, 40 to 137.996201, 50 fc rgb "#008080"
set object 59 rect from 0.115154, 40 to 156.197168, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.125900, 50 to 155.842036, 60 fc rgb "#FF0000"
set object 61 rect from 0.063473, 50 to 78.540109, 60 fc rgb "#00FF00"
set object 62 rect from 0.065907, 50 to 79.509548, 60 fc rgb "#0000FF"
set object 63 rect from 0.066384, 50 to 80.531778, 60 fc rgb "#FFFF00"
set object 64 rect from 0.067252, 50 to 82.054326, 60 fc rgb "#FF00FF"
set object 65 rect from 0.068529, 50 to 83.546879, 60 fc rgb "#808080"
set object 66 rect from 0.069762, 50 to 84.332747, 60 fc rgb "#800080"
set object 67 rect from 0.070670, 50 to 85.231396, 60 fc rgb "#008080"
set object 68 rect from 0.071156, 50 to 150.286953, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
