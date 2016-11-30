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

set object 15 rect from 0.211987, 0 to 145.513422, 10 fc rgb "#FF0000"
set object 16 rect from 0.200423, 0 to 136.376217, 10 fc rgb "#00FF00"
set object 17 rect from 0.202587, 0 to 136.885411, 10 fc rgb "#0000FF"
set object 18 rect from 0.203083, 0 to 137.445196, 10 fc rgb "#FFFF00"
set object 19 rect from 0.203926, 0 to 138.222811, 10 fc rgb "#FF00FF"
set object 20 rect from 0.205087, 0 to 139.039550, 10 fc rgb "#808080"
set object 21 rect from 0.206290, 0 to 139.433412, 10 fc rgb "#800080"
set object 22 rect from 0.207137, 0 to 139.808401, 10 fc rgb "#008080"
set object 23 rect from 0.207439, 0 to 142.267381, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.221054, 10 to 151.591403, 20 fc rgb "#FF0000"
set object 25 rect from 0.142556, 10 to 97.286136, 20 fc rgb "#00FF00"
set object 26 rect from 0.144611, 10 to 97.845921, 20 fc rgb "#0000FF"
set object 27 rect from 0.145198, 10 to 98.334875, 20 fc rgb "#FFFF00"
set object 28 rect from 0.145983, 10 to 99.161734, 20 fc rgb "#FF00FF"
set object 29 rect from 0.147160, 10 to 99.909008, 20 fc rgb "#808080"
set object 30 rect from 0.148272, 10 to 100.331874, 20 fc rgb "#800080"
set object 31 rect from 0.149156, 10 to 100.739234, 20 fc rgb "#008080"
set object 32 rect from 0.149499, 10 to 148.693366, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.219189, 20 to 150.424639, 30 fc rgb "#FF0000"
set object 34 rect from 0.126007, 20 to 86.143850, 30 fc rgb "#00FF00"
set object 35 rect from 0.128116, 20 to 86.609207, 30 fc rgb "#0000FF"
set object 36 rect from 0.128540, 20 to 87.224286, 30 fc rgb "#FFFF00"
set object 37 rect from 0.129451, 20 to 88.131393, 30 fc rgb "#FF00FF"
set object 38 rect from 0.130800, 20 to 88.863834, 30 fc rgb "#808080"
set object 39 rect from 0.131883, 20 to 89.248933, 30 fc rgb "#800080"
set object 40 rect from 0.132731, 20 to 89.707546, 30 fc rgb "#008080"
set object 41 rect from 0.133152, 20 to 147.437571, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.217434, 30 to 149.282818, 40 fc rgb "#FF0000"
set object 43 rect from 0.075894, 30 to 52.493063, 40 fc rgb "#00FF00"
set object 44 rect from 0.078195, 30 to 53.079822, 40 fc rgb "#0000FF"
set object 45 rect from 0.078822, 30 to 53.700298, 40 fc rgb "#FFFF00"
set object 46 rect from 0.079744, 30 to 54.587175, 40 fc rgb "#FF00FF"
set object 47 rect from 0.081072, 30 to 55.211027, 40 fc rgb "#808080"
set object 48 rect from 0.082006, 30 to 55.643330, 40 fc rgb "#800080"
set object 49 rect from 0.082885, 30 to 77.478745, 40 fc rgb "#008080"
set object 50 rect from 0.115080, 30 to 146.144692, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.215416, 40 to 149.533692, 50 fc rgb "#FF0000"
set object 52 rect from 0.051058, 40 to 36.343787, 50 fc rgb "#00FF00"
set object 53 rect from 0.054372, 40 to 37.155129, 50 fc rgb "#0000FF"
set object 54 rect from 0.055228, 40 to 38.752859, 50 fc rgb "#FFFF00"
set object 55 rect from 0.057621, 40 to 39.945250, 50 fc rgb "#FF00FF"
set object 56 rect from 0.059363, 40 to 40.728945, 50 fc rgb "#808080"
set object 57 rect from 0.060542, 40 to 41.223979, 50 fc rgb "#800080"
set object 58 rect from 0.061702, 40 to 42.071069, 50 fc rgb "#008080"
set object 59 rect from 0.062522, 40 to 144.780318, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.222580, 50 to 169.195431, 60 fc rgb "#FF0000"
set object 61 rect from 0.156495, 50 to 106.517086, 60 fc rgb "#00FF00"
set object 62 rect from 0.158304, 50 to 106.933209, 60 fc rgb "#0000FF"
set object 63 rect from 0.158677, 50 to 107.621813, 60 fc rgb "#FFFF00"
set object 64 rect from 0.159708, 50 to 108.559943, 60 fc rgb "#FF00FF"
set object 65 rect from 0.161106, 50 to 125.507075, 60 fc rgb "#808080"
set object 66 rect from 0.186282, 50 to 126.162615, 60 fc rgb "#800080"
set object 67 rect from 0.187531, 50 to 126.884262, 60 fc rgb "#008080"
set object 68 rect from 0.188263, 50 to 149.768416, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
