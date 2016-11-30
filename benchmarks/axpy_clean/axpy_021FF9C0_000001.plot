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

set object 15 rect from 0.139713, 0 to 158.647153, 10 fc rgb "#FF0000"
set object 16 rect from 0.119833, 0 to 134.579853, 10 fc rgb "#00FF00"
set object 17 rect from 0.121694, 0 to 135.292023, 10 fc rgb "#0000FF"
set object 18 rect from 0.122086, 0 to 136.157276, 10 fc rgb "#FFFF00"
set object 19 rect from 0.122882, 0 to 137.305395, 10 fc rgb "#FF00FF"
set object 20 rect from 0.123899, 0 to 138.496781, 10 fc rgb "#808080"
set object 21 rect from 0.124974, 0 to 139.090259, 10 fc rgb "#800080"
set object 22 rect from 0.125780, 0 to 139.693713, 10 fc rgb "#008080"
set object 23 rect from 0.126053, 0 to 154.411861, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.138235, 10 to 157.309346, 20 fc rgb "#FF0000"
set object 25 rect from 0.105224, 10 to 118.563834, 20 fc rgb "#00FF00"
set object 26 rect from 0.107248, 10 to 119.304840, 20 fc rgb "#0000FF"
set object 27 rect from 0.107670, 10 to 120.130157, 20 fc rgb "#FFFF00"
set object 28 rect from 0.108442, 10 to 121.475733, 20 fc rgb "#FF00FF"
set object 29 rect from 0.109654, 10 to 122.734784, 20 fc rgb "#808080"
set object 30 rect from 0.110801, 10 to 123.438077, 20 fc rgb "#800080"
set object 31 rect from 0.111696, 10 to 124.158016, 20 fc rgb "#008080"
set object 32 rect from 0.112048, 10 to 152.663611, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.136571, 20 to 155.223846, 30 fc rgb "#FF0000"
set object 34 rect from 0.092307, 20 to 103.975469, 30 fc rgb "#00FF00"
set object 35 rect from 0.094093, 20 to 104.596676, 30 fc rgb "#0000FF"
set object 36 rect from 0.094411, 20 to 105.512956, 30 fc rgb "#FFFF00"
set object 37 rect from 0.095238, 20 to 106.846333, 30 fc rgb "#FF00FF"
set object 38 rect from 0.096444, 20 to 107.887959, 30 fc rgb "#808080"
set object 39 rect from 0.097383, 20 to 108.518042, 30 fc rgb "#800080"
set object 40 rect from 0.098283, 20 to 109.325614, 30 fc rgb "#008080"
set object 41 rect from 0.098677, 20 to 150.889847, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.135005, 30 to 153.608744, 40 fc rgb "#FF0000"
set object 43 rect from 0.077371, 30 to 87.583397, 40 fc rgb "#00FF00"
set object 44 rect from 0.079337, 30 to 88.229010, 40 fc rgb "#0000FF"
set object 45 rect from 0.079655, 30 to 89.196318, 40 fc rgb "#FFFF00"
set object 46 rect from 0.080542, 30 to 90.557423, 40 fc rgb "#FF00FF"
set object 47 rect from 0.081766, 30 to 91.603487, 40 fc rgb "#808080"
set object 48 rect from 0.082697, 30 to 92.253539, 40 fc rgb "#800080"
set object 49 rect from 0.083588, 30 to 93.006752, 40 fc rgb "#008080"
set object 50 rect from 0.083963, 30 to 149.184855, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.133451, 40 to 152.053487, 50 fc rgb "#FF0000"
set object 52 rect from 0.063226, 40 to 72.344989, 50 fc rgb "#00FF00"
set object 53 rect from 0.065624, 40 to 73.073797, 50 fc rgb "#0000FF"
set object 54 rect from 0.065993, 40 to 74.046650, 50 fc rgb "#FFFF00"
set object 55 rect from 0.066874, 40 to 75.421071, 50 fc rgb "#FF00FF"
set object 56 rect from 0.068123, 40 to 76.491541, 50 fc rgb "#808080"
set object 57 rect from 0.069101, 40 to 77.220349, 50 fc rgb "#800080"
set object 58 rect from 0.070036, 40 to 78.023482, 50 fc rgb "#008080"
set object 59 rect from 0.070479, 40 to 147.327888, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.131694, 50 to 152.694662, 60 fc rgb "#FF0000"
set object 61 rect from 0.042156, 50 to 49.838475, 60 fc rgb "#00FF00"
set object 62 rect from 0.045511, 50 to 51.222872, 60 fc rgb "#0000FF"
set object 63 rect from 0.046307, 50 to 53.542417, 60 fc rgb "#FFFF00"
set object 64 rect from 0.048428, 50 to 55.459279, 60 fc rgb "#FF00FF"
set object 65 rect from 0.050133, 50 to 56.777126, 60 fc rgb "#808080"
set object 66 rect from 0.051429, 50 to 57.693398, 60 fc rgb "#800080"
set object 67 rect from 0.052669, 50 to 59.278590, 60 fc rgb "#008080"
set object 68 rect from 0.053572, 50 to 145.100422, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
