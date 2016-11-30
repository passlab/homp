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

set object 15 rect from 0.322544, 0 to 157.339488, 10 fc rgb "#FF0000"
set object 16 rect from 0.090634, 0 to 42.139217, 10 fc rgb "#00FF00"
set object 17 rect from 0.093450, 0 to 42.549509, 10 fc rgb "#0000FF"
set object 18 rect from 0.094076, 0 to 42.922668, 10 fc rgb "#FFFF00"
set object 19 rect from 0.094927, 0 to 43.482404, 10 fc rgb "#FF00FF"
set object 20 rect from 0.096169, 0 to 53.520533, 10 fc rgb "#808080"
set object 21 rect from 0.118336, 0 to 53.798590, 10 fc rgb "#800080"
set object 22 rect from 0.119215, 0 to 54.085704, 10 fc rgb "#008080"
set object 23 rect from 0.119566, 0 to 145.732195, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.329490, 10 to 160.300750, 20 fc rgb "#FF0000"
set object 25 rect from 0.284811, 10 to 129.829963, 20 fc rgb "#00FF00"
set object 26 rect from 0.287056, 10 to 130.147419, 20 fc rgb "#0000FF"
set object 27 rect from 0.287510, 10 to 130.523294, 20 fc rgb "#FFFF00"
set object 28 rect from 0.288346, 10 to 131.006950, 20 fc rgb "#FF00FF"
set object 29 rect from 0.289411, 10 to 141.007491, 20 fc rgb "#808080"
set object 30 rect from 0.311508, 10 to 141.263811, 20 fc rgb "#800080"
set object 31 rect from 0.312331, 10 to 141.529640, 20 fc rgb "#008080"
set object 32 rect from 0.312641, 10 to 148.959739, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.320354, 20 to 163.879715, 30 fc rgb "#FF0000"
set object 34 rect from 0.032141, 20 to 16.331092, 30 fc rgb "#00FF00"
set object 35 rect from 0.036584, 20 to 16.916189, 30 fc rgb "#0000FF"
set object 36 rect from 0.037493, 20 to 19.387455, 30 fc rgb "#FFFF00"
set object 37 rect from 0.042968, 20 to 25.092604, 30 fc rgb "#FF00FF"
set object 38 rect from 0.055540, 20 to 35.090428, 30 fc rgb "#808080"
set object 39 rect from 0.077721, 20 to 35.611671, 30 fc rgb "#800080"
set object 40 rect from 0.079166, 20 to 36.184541, 30 fc rgb "#008080"
set object 41 rect from 0.080042, 20 to 144.554755, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.327886, 30 to 167.839989, 40 fc rgb "#FF0000"
set object 43 rect from 0.230096, 30 to 104.971489, 40 fc rgb "#00FF00"
set object 44 rect from 0.232152, 30 to 105.296190, 40 fc rgb "#0000FF"
set object 45 rect from 0.232627, 30 to 106.805578, 40 fc rgb "#FFFF00"
set object 46 rect from 0.235976, 30 to 114.304512, 40 fc rgb "#FF00FF"
set object 47 rect from 0.252527, 30 to 124.262936, 40 fc rgb "#808080"
set object 48 rect from 0.274522, 30 to 124.657379, 40 fc rgb "#800080"
set object 49 rect from 0.275657, 30 to 125.040499, 40 fc rgb "#008080"
set object 50 rect from 0.276237, 30 to 148.270937, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.326386, 40 to 164.995566, 50 fc rgb "#FF0000"
set object 52 rect from 0.180310, 40 to 82.491443, 50 fc rgb "#00FF00"
set object 53 rect from 0.182531, 40 to 82.803011, 50 fc rgb "#0000FF"
set object 54 rect from 0.182960, 40 to 84.343646, 50 fc rgb "#FFFF00"
set object 55 rect from 0.186368, 40 to 89.680620, 50 fc rgb "#FF00FF"
set object 56 rect from 0.198151, 40 to 99.604174, 50 fc rgb "#808080"
set object 57 rect from 0.220070, 40 to 100.018542, 50 fc rgb "#800080"
set object 58 rect from 0.221246, 40 to 100.302486, 50 fc rgb "#008080"
set object 59 rect from 0.221608, 40 to 147.558586, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.324634, 50 to 164.140109, 60 fc rgb "#FF0000"
set object 61 rect from 0.128322, 50 to 58.996534, 60 fc rgb "#00FF00"
set object 62 rect from 0.130635, 50 to 59.386901, 60 fc rgb "#0000FF"
set object 63 rect from 0.131263, 50 to 60.855078, 60 fc rgb "#FFFF00"
set object 64 rect from 0.134508, 50 to 66.189787, 60 fc rgb "#FF00FF"
set object 65 rect from 0.146303, 50 to 76.039978, 60 fc rgb "#808080"
set object 66 rect from 0.168058, 50 to 76.482423, 60 fc rgb "#800080"
set object 67 rect from 0.169277, 50 to 76.802144, 60 fc rgb "#008080"
set object 68 rect from 0.169720, 50 to 146.660562, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
