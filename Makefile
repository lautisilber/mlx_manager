PREFIX ?= $(HOME)/.local
BINDIR := $(PREFIX)/bin
SCRIPT := mlx
SUPPORT_DIR := mlx_manager
TARGET := $(BINDIR)/$(SCRIPT)
SUPPORT_TARGET_DIR := $(BINDIR)/$(SUPPORT_DIR)

.PHONY: install uninstall

install:
	mkdir -p $(BINDIR) $(SUPPORT_TARGET_DIR)
	install -m 755 $(SCRIPT) $(TARGET)
	install -m 644 $(SUPPORT_DIR)/pack_*.py $(SUPPORT_TARGET_DIR)/
	@echo "Installed $(SCRIPT) -> $(TARGET)"
	@echo "Installed $(SUPPORT_DIR)/pack_*.py -> $(SUPPORT_TARGET_DIR)/"

uninstall:
	rm -f $(TARGET)
	rm -rf $(SUPPORT_TARGET_DIR)
	@echo "Removed $(TARGET)"
	@echo "Removed $(SUPPORT_TARGET_DIR)"
