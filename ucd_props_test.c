
#include "ucd_props.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  char buf[255];
  char *s = NULL, *ps = NULL;
  size_t len = 0;
  unsigned long cp = 0;
  const struct ucd_prop *prop = NULL;

  while (1) {
    memset(buf, 0, sizeof(buf));
    fputs("> ", stdout);
    s = fgets(buf, sizeof(buf), stdin);
    len = strlen(s);

    if (len == 1) {
      continue;
    }
    len -= 1;
    s[len] = 0;

    if (strcmp(s, "quit") == 0 || strcmp(s, "exit") == 0) {
      break;
    }

    ps = NULL;
    if (s[0] == '0') {
      if (len >= 2 && (s[1] == 'x' || s[1] == 'X')) {
        s = s + 2;
        cp = strtoul(s, &ps, 16);
      } else {
        s = s + 1;
        cp = strtoul(s, &ps, 8);
      }
    } else if (s[0] >= '1' && s[0] <= '9') {
      cp = strtoul(s, &ps, 10);
    }

    if (s == ps) {
      continue;
    }

    printf(".. cp=%d\n..   =0x%X\n", cp, cp);
    printf(".. index=%d\n", ucd_get_prop_index(cp));

    prop = ucd_get_prop(cp);

    printf(".. flags=%d\n.. extra_case=%d\n", prop->flags, 0 != (prop->flags & UCD_PROP_FLAG_EXTRA_CASE));

    printf(".. upper=%d\n.. lower=%d\n.. title=%d\n", cp + prop->upper,
           cp + prop->lower, cp + prop->title);
  }

  return 0;
}
