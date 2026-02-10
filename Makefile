.PHONY: build run clean install deps

# Binary name
BINARY_NAME=thales-fictitious-generator

# Build the application
build:
	@echo "Building $(BINARY_NAME)..."
	@go build -o $(BINARY_NAME) cmd/main.go

# Run the application
run: build
	@./$(BINARY_NAME)

# Install dependencies
deps:
	@echo "Installing dependencies..."
	@go mod download
	@go mod tidy

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -f $(BINARY_NAME)
	@go clean

install: build
	@echo "Installing $(BINARY_NAME) to /usr/local/bin..."
	@sudo mv $(BINARY_NAME) /usr/local/bin/
dev:
	@go run cmd/main.go

.DEFAULT_GOAL := build
