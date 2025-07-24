# Development Quality Assurance

This document describes the quality assurance tools and processes for RunNX development.

## Pre-commit Hook

A pre-commit hook has been installed that automatically runs the following checks before each commit:

1. **Code Formatting** - Ensures code follows consistent style using `cargo fmt --check`
2. **Linting** - Checks for common issues using `cargo clippy --all-targets --all-features -- -D warnings`
3. **Testing** - Runs all tests to ensure functionality remains intact
4. **Build Check** - Verifies the project builds successfully
5. **TODO/FIXME Scan** - Optionally warns about TODO/FIXME comments in staged files

### How it Works

The pre-commit hook is located at `.git/hooks/pre-commit` and runs automatically when you execute `git commit`.

If any check fails, the commit will be blocked and you'll need to fix the issues before committing.

### Manual Quality Check

You can manually run the same quality checks using:

```bash
./scripts/quality-check.sh
```

This script will:
- Format your code automatically
- Apply clippy fixes where possible
- Run all quality checks

## Code Formatting

RunNX uses `rustfmt` for consistent code formatting. The configuration is in `rustfmt.toml`.

### Key formatting rules:
- Maximum line width: 100 characters
- Use spaces (4 spaces per tab)
- Reorder imports and modules
- Use field init shorthand
- Conservative formatting that works with stable Rust

### Commands:
```bash
# Format code
cargo fmt

# Check formatting without changing files
cargo fmt --check
```

## Linting

We use `clippy` with strict settings to catch potential issues:

```bash
# Run clippy
cargo clippy --all-targets --all-features

# Run clippy treating warnings as errors
cargo clippy --all-targets --all-features -- -D warnings

# Auto-fix issues where possible
cargo clippy --fix
```

## Testing

Run the full test suite:

```bash
# Run all tests
cargo test

# Run with verbose output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

## Continuous Integration

The same checks that run in the pre-commit hook should be run in your CI/CD pipeline:

```bash
#!/bin/bash
set -e

echo "Running CI quality checks..."

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings

# Run tests
cargo test

# Build project
cargo build --all-targets

echo "All CI checks passed!"
```

## Bypassing Pre-commit Hook

In rare cases where you need to bypass the pre-commit hook:

```bash
git commit --no-verify -m "Your commit message"
```

**Warning**: Only use this for emergency fixes. The bypassed commit should be fixed in a follow-up commit.

## Troubleshooting

### Hook not running
- Ensure the hook file is executable: `chmod +x .git/hooks/pre-commit`
- Check that you're in the correct repository directory

### Formatting issues
- Run `cargo fmt` to fix formatting
- Check your `rustfmt.toml` configuration

### Test failures
- Run tests individually to isolate issues: `cargo test test_name`
- Use `cargo test -- --nocapture` for detailed output

### Build failures
- Check for compilation errors: `cargo build`
- Update dependencies if needed: `cargo update`

## Best Practices

1. **Commit early and often** - Small commits are easier to review and debug
2. **Write descriptive commit messages** - Help future maintainers understand changes
3. **Run quality checks locally** - Use `./scripts/quality-check.sh` before committing
4. **Keep tests updated** - Add tests for new features and bug fixes
5. **Address all clippy warnings** - Don't ignore linting suggestions

## Configuration Files

- `rustfmt.toml` - Code formatting configuration
- `.git/hooks/pre-commit` - Pre-commit hook script
- `scripts/quality-check.sh` - Manual quality check script
