clippy:
	cargo clippy --all

run:
	cargo run --release

test:
	cargo test --release -- --nocapture

build:
	RUSTFLAGS='-g' cargo build --release

profile: build
	valgrind --tool=callgrind --callgrind-out-file=callgrind.out		\
        --collect-jumps=yes --simulate-cache=yes target/release/dnnosaur

woods:
	RUSTFLAGS='-C target-feature=+crt-static' cargo build --release	\
	--target x86_64-unknown-linux-gnu
	scp -C target/x86_64-unknown-linux-gnu/release/dnnosaur 'woods:bin/.'
