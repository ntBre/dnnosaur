clippy:
	cargo clippy --all

build:
	RUSTFLAGS='-g' cargo build --release

profile: build
	valgrind --tool=callgrind --callgrind-out-file=callgrind.out		\
        --collect-jumps=yes --simulate-cache=yes target/release/dnnosaur
