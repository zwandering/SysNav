#CFLAGS+=-Wall -Werror
CFLAGS += -g -Og
OBJDIR=./obj
DEPDIR=./.depend
SRC = gstthetauvc.c gstthetauvcsrc.c thetauvc.c
OBJ := $(addprefix $(OBJDIR)/,$(SRC:.c=.o))
CFLAGS += $(shell pkg-config --cflags gstreamer-1.0 gstreamer-base-1.0 libuvc)
LDFLAGS += $(shell pkg-config --libs gstreamer-1.0 gstreamer-base-1.0 libuvc)

vpath %.o $(OBJDIR)

all: $(OBJDIR) gstthetauvc.so

install:
	cp gstthetauvc.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0

$(OBJDIR):
	mkdir $(OBJDIR)

$(OBJDIR)/%.o: %.c
	$(CC) -fPIC $(CFLAGS) -c -o $@ $<

gstthetauvc.so: $(OBJ)
	$(CC) -shared -o $@ $^  $(LDFLAGS)


.PHONY: clean depend
clean:
	$(RM) -r $(OBJDIR) *.o *.so

mostlyclean: clean
	$(RM) -r *~ $(DEPDIR)

$(DEPDIR):
	mkdir $(DEPDIR)

depend: $(DEPDIR)
	for f in $(SRC); do \
		$(CC) $(CFLAGS) -c -MM -o $(DEPDIR)/$${f%.c}.d -MT $(OBJDIR)/$${f%.c}.o $$f ;\
	done

-include $(DEPDIR)/*.d
