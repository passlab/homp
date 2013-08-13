#include <sys/timeb.h>
/* read timer in second */
double read_timer()
{
	struct timeb tm;
	ftime(&tm);
	return (double)tm.time + (double)tm.millitm/1000.0;
}

/* read timer in ms */
double read_timer_ms()
{
	struct timeb tm;
	ftime(&tm);
	return (double)tm.time * 1000.0 + (double)tm.millitm;
}
