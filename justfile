export RUST_BACKTRACE := "1"
alias c:= clean
alias f:= format
alias l:= clippy  # l for lint

@clean:
    rm -rf target  dist 

@coverage:
    cargo +nightly tarpaulin --verbose --all-features --workspace --timeout 120 --out html


@format:
     cargo fmt

@clippy:
    cargo clippy -- -D warnings -A incomplete_features -W clippy::dbg_macro -W clippy::print_stdout