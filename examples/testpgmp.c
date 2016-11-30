#include <stdlib.h>
#include <stdio.h>
#include "addresstranslation.h"

int main(int argc, char* argv[])
{
  int *a1, *a2, *b1,*c1, *b2 = NULL;
   int N = 4096;
   uintptr_t ap1, ap2, bp1, bp2 = 0;

   printf("Test virtual to physical address translation.\n");

   a1 = (int*)malloc(sizeof(int) * N);
   if (!a1)
   {
      printf("Error: cannot allocate memory for a\n");
      return 1;
   }

   b1 = (int*)malloc(sizeof(int) * N);
   if (!b1)
   {
      printf("Error: cannot allocate memory for b\n");
      return 1;
   }

   ap1 = virtual_to_physical_address((uintptr_t)a1);
   bp1 = virtual_to_physical_address((uintptr_t)b1);

   printf("a1_virt= %p: a1_phys= %" PRIxPTR "\n", a1, ap1);
   printf("b1_virt= %p b1_phys= %" PRIxPTR "\n", b1, bp1);
   a2 = a1 + 1000;
   b2 = b1 + 1;
   ap2 = virtual_to_physical_address((uintptr_t)a2);
   bp2 = virtual_to_physical_address((uintptr_t)b2);
   printf("a2_virt= %p a2_phys= %" PRIxPTR "\n", a2, ap2);
   printf("b2_virt= %p b2_phys= %" PRIxPTR "\n", b2, bp2);
   printf("Done\n");
}
