.PHONY: build server dev cli test bench fmt lint clean

PORT ?= 8080
HOST ?= 0.0.0.0
BIN := target/release/automl

build:
	cargo build --release

server: build
	$(BIN) serve --host $(HOST) --port $(PORT)

dev:
	cargo run -- serve --host $(HOST) --port $(PORT)

cli:
	cargo run -- --help

test:
	cargo test

bench:
	cargo bench

fmt:
	cargo fmt

lint:
	cargo clippy --all-targets -- -D warnings

clean:
	cargo clean
