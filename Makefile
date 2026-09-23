PREFIX ?= $(HOME)/.local
BINDIR := $(PREFIX)/bin
SCRIPT := mlx
TARGET := $(BINDIR)/$(SCRIPT)

.PHONY: install uninstall

install:
	mkdir -p $(BINDIR)
	install -m 755 $(SCRIPT) $(TARGET)
	@echo "Installed $(SCRIPT) -> $(TARGET)"

uninstall:
	rm -f $(TARGET)
	@echo "Removed $(TARGET)"
