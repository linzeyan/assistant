# Assistant — local-first AI assistant (Python backend + SwiftUI app)
#
# Quick start after clone:
#   make setup      # create venv + install python deps
#   make test       # run the test suite (no omlx needed)
#   make run        # start the backend
#   make app-run    # build + launch the SwiftUI app (separate terminal)
#
# Inference backend (config `model_backend`): defaults to "mlx" (native, in-process
# mlx-lm — `make setup-mlx`). "omlx" uses an external omlx server (`make omlx`).
# Neither is needed to set up, test, or develop — only to actually run inference.

UV      := uv
VENV    := .venv
PY      := $(VENV)/bin/python
APP_DIR := apps/assistant-mac
DIST    := dist
APP     := Assistant

.DEFAULT_GOAL := help
.PHONY: help install setup setup-mlx setup-images setup-embeddings setup-vlm setup-audio setup-video test run backend app-build app-test app-run app-package app-notarize omlx clean

help:  ## show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## install the APP: venv + ALL managed MLX tools (the way to run it)
	@test -d $(VENV) || $(UV) venv
	$(UV) pip install -e ".[all]"

setup:  ## create venv (if needed) + install python dev deps only  [for development]
	@test -d $(VENV) || $(UV) venv
	$(UV) pip install -e ".[dev]"

setup-mlx:  ## dev + mlx-lm (native, no-omlx inference backend)
	@test -d $(VENV) || $(UV) venv
	$(UV) pip install -e ".[dev,mlx]"

setup-images:  ## dev + mflux (image generation, Apple Silicon)
	@test -d $(VENV) || $(UV) venv
	$(UV) pip install -e ".[dev,images]"

setup-embeddings:  ## dev + mlx-embeddings (semantic memory search)
	@test -d $(VENV) || $(UV) venv
	$(UV) pip install -e ".[dev,embeddings]"

setup-vlm:  ## dev + mlx-vlm (vision: read images)
	@test -d $(VENV) || $(UV) venv
	$(UV) pip install -e ".[dev,vlm]"

setup-audio:  ## dev + mlx-audio (speech-to-text + text-to-speech)
	@test -d $(VENV) || $(UV) venv
	$(UV) pip install -e ".[dev,audio]"

setup-video:  ## dev + mlx-video (text-to-video: Wan / LTX)
	@test -d $(VENV) || $(UV) venv
	$(UV) pip install -e ".[dev,video]"

test:  ## run the python test suite (mocks omlx; no install required)
	$(PY) -m pytest -q

run:  ## start the backend (native MLX by default; degrades gracefully if absent)
	$(VENV)/bin/assistant-server

backend: run  ## alias for `run`

app-build:  ## compile the SwiftUI app (debug)
	cd $(APP_DIR) && swift build

# swift-testing's Testing.framework ships with Command Line Tools but isn't on the
# default search path (this project is CLT-only by design — see Package.swift), so point
# the compiler/linker at it. Paths derive from `xcode-select -p`; under a full Xcode
# toolchain `swift test` resolves Testing on its own and these extra flags are ignored.
app-test:  ## run the SwiftUI app's Swift unit tests (local; not wired into CI)
	cd $(APP_DIR) && \
	  FWK="$$(xcode-select -p)/Library/Developer/Frameworks" && \
	  LIB="$$(xcode-select -p)/Library/Developer/usr/lib" && \
	  swift test \
	    -Xswiftc -F -Xswiftc "$$FWK" \
	    -Xlinker -F -Xlinker "$$FWK" \
	    -Xlinker -rpath -Xlinker "$$FWK" \
	    -Xlinker -rpath -Xlinker "$$LIB"

app-run:  ## build + launch the SwiftUI app
	cd $(APP_DIR) && swift run

app-package:  ## build + bundle + code-sign dist/Assistant.app (ad-hoc; CODESIGN_IDENTITY=... for Developer ID)
	bash $(APP_DIR)/scripts/package.sh

app-notarize:  ## notarize + staple the signed .app (needs Developer ID build + NOTARY_PROFILE)
	cd $(DIST) && ditto -c -k --keepParent $(APP).app $(APP).zip
	xcrun notarytool submit $(DIST)/$(APP).zip --keychain-profile "$(NOTARY_PROFILE)" --wait
	xcrun stapler staple $(DIST)/$(APP).app
	@echo "Notarized + stapled $(DIST)/$(APP).app"

omlx:  ## install the omlx model server (Homebrew; needed only for inference)
	brew install omlx

clean:  ## remove venv, build artifacts, caches, and packaging output
	rm -rf $(VENV) $(DIST) $(APP_DIR)/.build .pytest_cache *.egg-info assistant.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
