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

set object 15 rect from 0.277718, 0 to 158.645650, 10 fc rgb "#FF0000"
set object 16 rect from 0.243439, 0 to 131.829659, 10 fc rgb "#00FF00"
set object 17 rect from 0.245216, 0 to 132.131132, 10 fc rgb "#0000FF"
set object 18 rect from 0.245542, 0 to 132.447141, 10 fc rgb "#FFFF00"
set object 19 rect from 0.246145, 0 to 132.946724, 10 fc rgb "#FF00FF"
set object 20 rect from 0.247063, 0 to 141.073569, 10 fc rgb "#808080"
set object 21 rect from 0.262175, 0 to 141.339504, 10 fc rgb "#800080"
set object 22 rect from 0.262866, 0 to 141.581222, 10 fc rgb "#008080"
set object 23 rect from 0.263094, 0 to 149.263394, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.270124, 10 to 156.016921, 20 fc rgb "#FF0000"
set object 25 rect from 0.040610, 10 to 23.631686, 20 fc rgb "#00FF00"
set object 26 rect from 0.044437, 10 to 24.401512, 20 fc rgb "#0000FF"
set object 27 rect from 0.045435, 10 to 25.378072, 20 fc rgb "#FFFF00"
set object 28 rect from 0.047263, 10 to 26.118827, 20 fc rgb "#FF00FF"
set object 29 rect from 0.048623, 10 to 34.388334, 20 fc rgb "#808080"
set object 30 rect from 0.064031, 10 to 34.725336, 20 fc rgb "#800080"
set object 31 rect from 0.064962, 10 to 35.079569, 20 fc rgb "#008080"
set object 32 rect from 0.065285, 10 to 144.676162, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.271965, 20 to 163.313614, 30 fc rgb "#FF0000"
set object 34 rect from 0.074398, 20 to 40.942668, 30 fc rgb "#00FF00"
set object 35 rect from 0.076370, 20 to 41.296901, 30 fc rgb "#0000FF"
set object 36 rect from 0.076812, 20 to 42.936155, 30 fc rgb "#FFFF00"
set object 37 rect from 0.079878, 20 to 49.566938, 30 fc rgb "#FF00FF"
set object 38 rect from 0.092206, 20 to 57.692163, 30 fc rgb "#808080"
set object 39 rect from 0.107294, 20 to 58.225125, 30 fc rgb "#800080"
set object 40 rect from 0.108567, 20 to 58.785538, 30 fc rgb "#008080"
set object 41 rect from 0.109311, 20 to 146.065091, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.275169, 30 to 163.694248, 40 fc rgb "#FF0000"
set object 43 rect from 0.164161, 30 to 89.014090, 40 fc rgb "#00FF00"
set object 44 rect from 0.165661, 30 to 89.308568, 40 fc rgb "#0000FF"
set object 45 rect from 0.165991, 30 to 90.857919, 40 fc rgb "#FFFF00"
set object 46 rect from 0.168870, 30 to 96.469081, 40 fc rgb "#FF00FF"
set object 47 rect from 0.179294, 30 to 104.416114, 40 fc rgb "#808080"
set object 48 rect from 0.194064, 30 to 104.902773, 40 fc rgb "#800080"
set object 49 rect from 0.195198, 30 to 105.210166, 40 fc rgb "#008080"
set object 50 rect from 0.195532, 30 to 147.852389, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.273689, 40 to 163.848735, 50 fc rgb "#FF0000"
set object 52 rect from 0.117564, 40 to 64.004250, 50 fc rgb "#00FF00"
set object 53 rect from 0.119196, 40 to 64.357400, 50 fc rgb "#0000FF"
set object 54 rect from 0.119644, 40 to 65.782933, 50 fc rgb "#FFFF00"
set object 55 rect from 0.122311, 40 to 72.390027, 50 fc rgb "#FF00FF"
set object 56 rect from 0.134567, 40 to 80.359674, 50 fc rgb "#808080"
set object 57 rect from 0.149379, 40 to 80.851724, 50 fc rgb "#800080"
set object 58 rect from 0.150506, 40 to 81.168270, 50 fc rgb "#008080"
set object 59 rect from 0.150890, 40 to 146.948516, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.276450, 50 to 164.953413, 60 fc rgb "#FF0000"
set object 61 rect from 0.203092, 50 to 109.972357, 60 fc rgb "#00FF00"
set object 62 rect from 0.204592, 50 to 110.288903, 60 fc rgb "#0000FF"
set object 63 rect from 0.204963, 50 to 111.791422, 60 fc rgb "#FFFF00"
set object 64 rect from 0.207785, 50 to 118.048054, 60 fc rgb "#FF00FF"
set object 65 rect from 0.219377, 50 to 125.970323, 60 fc rgb "#808080"
set object 66 rect from 0.234116, 50 to 126.453758, 60 fc rgb "#800080"
set object 67 rect from 0.235252, 50 to 126.761151, 60 fc rgb "#008080"
set object 68 rect from 0.235572, 50 to 148.587769, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
