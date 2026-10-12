.PHONY: project setup-sim build test lint fmt typecheck checkall run-sim screenshot run-phone  checkall-ci build-phone pre-commit pre-commit-update clean

APP_NAME ?= PhoneFluid
SCHEME ?= PhoneFluid
BUNDLE_ID ?= com.paulrobello.PhoneFluid
TEAM_ID ?= QMLVG482FY
SIMULATOR_NAME ?= PhoneFluid iPhone 18 Pro Max
SIMULATOR_OS ?= 27.0
SIMULATOR_DEVICE_TYPE ?= com.apple.CoreSimulator.SimDeviceType.iPhone-18-Pro-Max
SIMULATOR_RUNTIME ?= com.apple.CoreSimulator.SimRuntime.iOS-27-0
DESTINATION ?= platform=iOS Simulator,name=$(SIMULATOR_NAME),OS=$(SIMULATOR_OS)
PHONE_DEVICE ?= Pauls iPhone 18 Pro Max
PHONE_DESTINATION ?= platform=iOS,name=$(PHONE_DEVICE)
DERIVED_DATA ?= $(HOME)/Library/Developer/Xcode/DerivedData
SCREENSHOT_PATH ?= build/screenshot.png
FLUID_GRAVITY ?= 0,-1,0
FLUID_GRAVITY_SWEEP ?=
FLUID_PRESET ?=
FLUID_BACKDROP ?=
FLUID_STATS ?=
FLUID_QUALITY ?=
FORMAT_DIRS ?= PhoneFluid PhoneFluidTests

project:  ## Generate the Xcode project from project.yml
	xcodegen generate

setup-sim:  ## Create the dedicated simulator if it does not exist
	@if ! xcrun simctl list devices available | awk -v runtime="iOS $(SIMULATOR_OS)" -v name="$(SIMULATOR_NAME)" '\
		/^-- / { active = ($$0 == "-- " runtime " --"); next } \
		active && index($$0, name " (") { found = 1 } \
		END { exit found ? 0 : 1 }'; then \
		xcrun simctl create "$(SIMULATOR_NAME)" "$(SIMULATOR_DEVICE_TYPE)" "$(SIMULATOR_RUNTIME)" >/dev/null; \
	fi

build: project setup-sim  ## Generate the project and build for the simulator
	xcodebuild build -scheme $(SCHEME) -destination '$(DESTINATION)' -quiet

test: project setup-sim  ## Run the unit test bundle in the simulator
	xcodebuild test -scheme $(SCHEME) -destination '$(DESTINATION)' -quiet

lint:  ## Lint Swift sources with swift-format
	@if command -v swift-format >/dev/null 2>&1; then \
		swift-format lint --strict -r $(FORMAT_DIRS); \
	else \
		echo "swift-format not installed, skipping lint"; \
	fi

fmt:  ## Format Swift sources in place with swift-format
	@if command -v swift-format >/dev/null 2>&1; then \
		swift-format format -i -r $(FORMAT_DIRS); \
	else \
		echo "swift-format not installed, skipping format"; \
	fi

typecheck: project setup-sim  ## Compile-only simulator build with signing disabled
	xcodebuild build -scheme $(SCHEME) -destination '$(DESTINATION)' -quiet CODE_SIGNING_ALLOWED=NO

checkall: fmt lint typecheck test  ## Canonical local gate: format, lint, typecheck, test

checkall-ci: checkall  ## CI-parity gate (no CI configured; same as checkall)

run-sim: project setup-sim  ## Build, install, and launch on the simulator
	SIMULATOR_UDID="$$(xcrun simctl list devices available | awk -v runtime="iOS $(SIMULATOR_OS)" -v name="$(SIMULATOR_NAME)" '\
		/^-- / { active = ($$0 == "-- " runtime " --"); next } \
		active && index($$0, name " (") { id = $$0; sub("^.*" name " \\(", "", id); sub("\\).*$$", "", id); print id; exit }')"; \
	if [ -z "$$SIMULATOR_UDID" ]; then \
		printf 'No simulator found for %s on iOS %s\n' "$(SIMULATOR_NAME)" "$(SIMULATOR_OS)" >&2; \
		exit 1; \
	fi; \
	open -a Simulator; \
	xcrun simctl boot "$$SIMULATOR_UDID" 2>/dev/null || true; \
	xcrun simctl bootstatus "$$SIMULATOR_UDID" -b; \
	xcrun simctl ui "$$SIMULATOR_UDID" appearance dark; \
	xcodebuild build -scheme $(SCHEME) -destination '$(DESTINATION)' -quiet; \
	xcrun simctl install "$$SIMULATOR_UDID" "$$(ls -td "$(DERIVED_DATA)"/$(APP_NAME)-*/Build/Products/Debug-iphonesimulator/$(APP_NAME).app | head -1)"; \
	SIMCTL_CHILD_FLUID_GRAVITY="$(FLUID_GRAVITY)" SIMCTL_CHILD_FLUID_GRAVITY_SWEEP="$(FLUID_GRAVITY_SWEEP)" \
		SIMCTL_CHILD_FLUID_PRESET="$(FLUID_PRESET)" SIMCTL_CHILD_FLUID_BACKDROP="$(FLUID_BACKDROP)" \
		SIMCTL_CHILD_FLUID_STATS="$(FLUID_STATS)" SIMCTL_CHILD_FLUID_QUALITY="$(FLUID_QUALITY)" \
		xcrun simctl launch --terminate-running-process "$$SIMULATOR_UDID" $(BUNDLE_ID)

screenshot: run-sim  ## Capture a simulator screenshot to SCREENSHOT_PATH (agent operability)
	@mkdir -p "$(dir $(SCREENSHOT_PATH))"
	sleep $${SCREENSHOT_DELAY:-6}
	SIMULATOR_UDID="$$(xcrun simctl list devices available | awk -v runtime="iOS $(SIMULATOR_OS)" -v name="$(SIMULATOR_NAME)" '\
		/^-- / { active = ($$0 == "-- " runtime " --"); next } \
		active && index($$0, name " (") { id = $$0; sub("^.*" name " \\(", "", id); sub("\\).*$$", "", id); print id; exit }')"; \
	xcrun simctl io "$$SIMULATOR_UDID" screenshot "$(SCREENSHOT_PATH)"; \
	printf 'Saved simulator screenshot to %s\n' "$(SCREENSHOT_PATH)"

build-phone: project  ## Compile for a physical iPhone (generic device, no install)
	xcodebuild build -scheme $(SCHEME) -destination 'generic/platform=iOS' -allowProvisioningUpdates -quiet DEVELOPMENT_TEAM=$(TEAM_ID)

run-phone: project  ## Build, install, and launch on the connected iPhone (by name)
	xcodebuild build -scheme $(SCHEME) -destination '$(PHONE_DESTINATION)' -allowProvisioningUpdates -quiet DEVELOPMENT_TEAM=$(TEAM_ID)
	xcrun devicectl device install app --device "$(PHONE_DEVICE)" "$$(ls -td "$(DERIVED_DATA)"/$(APP_NAME)-*/Build/Products/Debug-iphoneos/$(APP_NAME).app | head -1)"
	xcrun devicectl device process launch --device "$(PHONE_DEVICE)" --terminate-existing $(BUNDLE_ID)

pre-commit:  ## Run all pre-commit hooks against the whole tree
	pre-commit run --all-files

pre-commit-update:  ## Update pinned pre-commit hook revisions
	pre-commit autoupdate

clean: project  ## Remove build products
	xcodebuild clean -scheme $(SCHEME) -destination '$(DESTINATION)' -quiet
