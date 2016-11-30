#include "addresstranslation.h"
#include <stdio.h>
#define PAGEMAP_ENTRY 8
#define GET_BIT(X,Y) (X & ((uint64_t)1<<Y)) >> Y
#define GET_PFN(X) X & 0x7FFFFFFFFFFFFF
#define page_mapping_file "/proc/self/pagemap"

const int __endian_bit = 1;
#define is_bigendian() ( (*(char*)&__endian_bit) == 0 )


uintptr_t virtual_to_physical_address(uintptr_t virt_addr)
{
   uintptr_t file_offset = 0;
   uintptr_t read_val = 0;
   uintptr_t page_number = 0;
   int i = 0;
   int c = 0;
   int pid = 0;
   int status = 0;
   unsigned char c_buf[PAGEMAP_ENTRY];

   FILE *f = fopen(page_mapping_file, "rb");
   if(!f)
   {
      // if this happens run as root
      printf("Error! Cannot open %s. Please, run as root.\n", page_mapping_file);
      return 0;
   }

   file_offset = virt_addr / getpagesize() * PAGEMAP_ENTRY;

   status = fseek(f, file_offset, SEEK_SET);
   if(status)
   {
      printf("Error! Cannot seek in %s.\n", page_mapping_file);
      perror("Failed to do fseek!");
      fclose(f);
      return 0;
   }

   for(i = 0; i < PAGEMAP_ENTRY; i++)
   {
      c = getc(f);
      if(c == EOF)
      {
         fclose(f);
         return 0;
      }

      if(is_bigendian())
      {
           c_buf[i] = c;
      }
      else
      {
           c_buf[PAGEMAP_ENTRY - i - 1] = c;
      }
   }

   for(i=0; i < PAGEMAP_ENTRY; i++)
   {
      read_val = (read_val << 8) + c_buf[i];
   }

   /*
   if(GET_BIT(read_val, 63))
   {
      page_number = GET_PFN(read_val);
      printf("%d \n", page_number);
   }
   else
   {
     printf("Page not present\n");
   }
   if(GET_BIT(read_val, 62))
   {
      printf("Page swapped\n");
   }
   */
   fclose(f);

   return read_val;
}
